import re
import json
import ast
from backend.agent import get_zai_client, get_references_tool, AgentState, safe_json_loads, normalize_doc_id
# 하이브리드 방식: 참조 조회는 Tool, 정밀 영향 분석은 Store 직접 호출
from backend.graph_store import Neo4jGraphStore


# Graph Agent를 고도화하여 다음과 같은 기능을 추가했습니다.
# 질문 의도 분석: 영향 분석(Impact), 의존성 분석(Dependency) 등을 구분하여 답변합니다.
# 시각화 지원: 관계를 한눈에 볼 수 있도록 Mermaid 다이어그램을 생성합니다.
# 전문 보고서 형식: 참조 관계의 의미와 변경 시 주의사항을 포함한 상세 보고서를 제공합니다.
# 이제 에이전트에게 "SOP-xxx 변경 시 영향 알려줘"와 같이 질문하면 더욱 풍부한 답변을 받을 수 있습니다.
def generate_mermaid_flow(doc_id: str, refs: dict, impact_data: list = None) -> str:
    """Mermaid 다이어그램 코드 생성 (참조 및 영향 분석 통합)"""
    lines = ["graph LR"]
    safe_doc_id = doc_id.replace("-", "_")
    
    doc = refs.get("document") or {}
    title = (doc.get("title") or doc_id).replace('"', "'")
    
    # 메인 노드 스타일
    lines.append(f'    Main["{doc_id}<br/>({title})"]:::mainNode')
    
    # 1. 문서 간 참조 (References/Cited By)
    for ref in refs.get("references", []):
        ref_id = ref.replace("-", "_")
        lines.append(f'    Main --> {ref_id}["{ref}"]')
        
    for cited in refs.get("cited_by", []):
        cited_id = cited.replace("-", "_")
        lines.append(f'    {cited_id}["{cited}"] --> Main')

    # 2. 영향 분석 데이터가 있는 경우 추가 (정밀 관계)
    if impact_data:
        for idx, imp in enumerate(impact_data):
            src_id = imp.get("source_doc_id", "Unknown").replace("-", "_")
            section = imp.get("citing_section", "")
            lines.append(f'    Main -- "{section} 조항에서 언급" --> {src_id}')

    lines.append("    classDef mainNode fill:#f96,stroke:#333,stroke-width:4px;")
    return "\n".join(lines)

def graph_agent_node(state: AgentState):
    """[서브] 그래프 에이전트 (Z.AI) - 인텐트 분석 및 시각화 지원"""
    client = get_zai_client()
    query = state["query"]
    model = state.get("worker_model") or state.get("model_name") or "glm-4.7-flash"
    
    # 1. 의도 및 엔티티 추출
    messages = state.get("messages", [])
    extraction_prompt = f"""사용자의 질문과 대화 이력을 분석하여 분석 대상이 되는 SOP ID와 질문의 의도를 추출하세요.
    - 질문: {query}
    
    [의도 분류]
    - impact_analysis: 특정 문서를 변경했을 때 영향을 받는 하위 문서나 관련 절차를 찾고자 할 때
    - dependency_analysis: 특정 문서가 작동하기 위해 참조해야 하는 상위 규정이나 근거를 찾고자 할 때
    - relationship_check: 두 문서 사이의 연결 고리를 확인하고자 할 때
    - general_info: 단순히 특정 문서의 참조 목록을 보고 싶어할 때
    
    반드시 JSON 형식으로만 답변하세요.
    예: {{"doc_id": "EQ-SOP-04", "intent": "impact_analysis"}}"""
    
    try:
        extraction_res = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": "엔티티 추출 전문가입니다."}] + messages + [{"role": "user", "content": extraction_prompt}],
            response_format={"type": "json_object"}
        )
        info = safe_json_loads(extraction_res.choices[0].message.content)
        doc_id = normalize_doc_id(info.get("doc_id"))
        intent = info.get("intent", "general_info")
    except:
        raw_match = re.search(r'([A-Za-z0-9_-]*SOP-\d+)', query.upper())
        doc_id = normalize_doc_id(raw_match.group(1)) if raw_match else None
        intent = "general_info"
    
    if not doc_id:
        return {"messages": [{"role": "assistant", "content": "[그래프 에이전트] 분석할 문서 ID를 찾지 못했습니다. 요청에 문서 번호를 포함해 주세요."}]}

    # 2. 데이터 조회
    refs_str = get_references_tool.invoke({"doc_id": doc_id})
    ref_data = safe_json_loads(refs_str) or {"document": {"doc_id": doc_id}, "references": [], "cited_by": []}
    
    impact_list = []
    if intent == "impact_analysis" or "영향" in query:
        try:
            graph_store = Neo4jGraphStore()
            graph_store.connect()
            impact_list = graph_store.get_impact_analysis(doc_id)
        except Exception as e:
            print(f"[DEBUG graph.py] Graph 직접 조회 실패: {e}")
            impact_list = []

    # 3. 시각화 (Mermaid) 생성
    mermaid_code = generate_mermaid_flow(doc_id, ref_data, impact_list)
    
    # 4. 심층 분석 (Z.AI)
    analysis_prompt = f"""다음 그래프 데이터를 바탕으로 질문에 대해 간단명료한 분석 보고서를 작성하세요.
    질문: {query}
    의도: {intent}
    데이터: {json.dumps(ref_data, ensure_ascii=False)}
    정밀 영향 분석: {json.dumps(impact_list, ensure_ascii=False)}

    [보고서 작성 규칙]
    - **FACTS ONLY**: 제공된 데이터(참조 관계, 영향 분석)에 없는 문서나 관계를 절대 언급하지 마세요.
    - **NO MARKDOWN EXCLUDING MERMAID**: 오직 머메이드 코드 블록( ```mermaid ) 외에는 마크다운 기호를 쓰지 마세요.
    - **PLAIN TEXT ONLY**: 줄글과 들여쓰기만 사용하세요.
    - 영향 분석 데이터가 없다면 "영향 분석 데이터가 없습니다"라고 명시하세요. 짐작해서 답변하지 마세요.

    [답변 형식]
    {doc_id} 관계 및 영향 분석 보고
    
    1. 참조 관계
    (참조하는/받는 문서 목록)
    
    2. 파급 효과 (영향 분석)
    (구체적인 영향 조항 및 내용 설명)
    
    3. 종합 의견
    (짧은 결론)
    """

    analysis_res = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": analysis_prompt}]
    )

    llm_analysis = analysis_res.choices[0].message.content.strip()
    
    final_report = f"""### [그래프 에이전트 관계 분석 보고]

{llm_analysis}

#### 관계 시각화 (Mermaid)
```mermaid
{mermaid_code}
```

[DONE]"""

    return {"context": [final_report]}
