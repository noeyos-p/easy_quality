import re
import json
import ast
from typing import List, Dict, Any, Optional, TypedDict
from backend.agent import get_zai_client, get_references_tool, AgentState

# ===============================
# State Definition
# ===============================
class GraphSubAgentState(TypedDict):
    query: str
    messages: List[Dict[str, str]]
    model: str

    sop_id: Optional[str]
    intent: str

    base_refs: Dict[str, Any]
    final_report: str


# ===============================
# Helper: Mermaid Graph Generator
# ===============================
def generate_mermaid_flow(sop_id: str, refs: dict) -> str:
    """Mermaid 다이어그램 코드 생성"""
    lines = ["graph LR"]

    doc = refs.get("document") or {}
    title = doc.get("title", "Unknown")

    safe_id = sop_id.replace("-", "_")
    lines.append(f'    Main["{sop_id}<br/>({title})"]:::mainNode')

    # 참조하는 문서들
    for ref in refs.get("references", []):
        ref_id = ref.replace("-", "_")
        lines.append(f'    Main --> {ref_id}["{ref}"]')

    # 참조되는 문서들
    for cited in refs.get("cited_by", []):
        cited_id = cited.replace("-", "_")
        lines.append(f'    {cited_id}["{cited}"] --> Main')

    lines.append("    classDef mainNode fill:#f96,stroke:#333,stroke-width:4px;")
    return "\n".join(lines)


# ===============================
# Main Graph Agent Node
# ===============================
def graph_agent_node(state: AgentState):
    """
    [서브] Graph Agent
    - 질문 의도 분석 (impact / dependency / relationship / general)
    - Neo4j 참조 관계 조회
    - Mermaid 시각화 + 전문 분석 리포트 생성
    """

    client = get_zai_client()
    query = state["query"]
    messages = state.get("messages", [])
    model = state.get("worker_model") or state.get("model_name") or "glm-4.7-flash"

    # -------------------------------
    # 1. SOP ID + Intent 추출
    # -------------------------------
    extraction_prompt = f"""
사용자의 질문과 대화 이력을 분석하여 분석 대상 SOP ID와 질문 의도를 추출하세요.

질문: {query}

[의도 분류]
- impact_analysis
- dependency_analysis
- relationship_check
- general_info

JSON 형식으로만 답변:
{{"sop_id": "EQ-SOP-001", "intent": "impact_analysis", "reason": "이유"}}
"""

    try:
        res = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": "당신은 SOP 관계 분석 전문가입니다."}]
                     + messages
                     + [{"role": "user", "content": extraction_prompt}],
            response_format={"type": "json_object"}
        )
        info = json.loads(res.choices[0].message.content)
        sop_id = info.get("sop_id")
        intent = info.get("intent", "general_info")
    except Exception:
        match = re.search(r'([A-Z0-9]+-SOP-\d+)', query.upper())
        sop_id = match.group(1) if match else None
        intent = "general_info"

    if not sop_id:
        return {
            "messages": [{
                "role": "assistant",
                "content": "[그래프 에이전트] 분석할 SOP ID를 찾지 못했습니다. (예: EQ-SOP-001 영향 분석)"
            }]
        }

    # -------------------------------
    # 2. Graph DB 조회
    # -------------------------------
    refs_str = get_references_tool.invoke({"sop_id": sop_id})

    if not refs_str or refs_str == "None":
        return {
            "messages": [{
                "role": "assistant",
                "content": f"[그래프 에이전트] {sop_id}에 대한 참조 데이터가 없습니다."
            }]
        }

    try:
        ref_data = ast.literal_eval(refs_str)
    except Exception:
        ref_data = {"document": {"sop_id": sop_id}, "references": [], "cited_by": []}

    # -------------------------------
    # 3. Mermaid 시각화
    # -------------------------------
    mermaid_code = generate_mermaid_flow(sop_id, ref_data)

    # -------------------------------
    # 4. 분석 리포트 생성
    # -------------------------------
    analysis_prompt = f"""
다음 그래프 데이터를 바탕으로 질문에 대한 분석 보고서를 작성하세요.

질문: {query}
의도: {intent}
데이터: {json.dumps(ref_data, ensure_ascii=False)}

작성 규칙:
- 의도에 맞게 영향 / 의존성 중심으로 설명
- 불렛 포인트 위주
- 한국어
"""

    analysis_res = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": analysis_prompt}]
    )

    final_report = f"""### 🧠 {sop_id} 관계 분석 보고서

{analysis_res.choices[0].message.content}

#### 🔗 관계 시각화 (Mermaid)
```mermaid
{mermaid_code}
