import re
import json
import ast
from backend.agent import get_zai_client, get_references_tool, AgentState, safe_json_loads, normalize_doc_id


# Graph Agent를 고도화하여 다음과 같은 기능을 추가했습니다.
# 질문 의도 분석: 영향 분석(Impact), 의존성 분석(Dependency) 등을 구분하여 답변합니다.
# 시각화 지원: 관계를 한눈에 볼 수 있도록 Mermaid 다이어그램을 생성합니다.
# 전문 보고서 형식: 참조 관계의 의미와 변경 시 주의사항을 포함한 상세 보고서를 제공합니다.
# 이제 에이전트에게 "SOP-xxx 변경 시 영향 알려줘"와 같이 질문하면 더욱 풍부한 답변을 받을 수 있습니다.
def generate_mermaid_flow(doc_id: str, refs: dict) -> str:
    """Mermaid 다이어그램 코드 생성"""
    lines = ["graph LR"]
    
    doc = refs.get("document") or {}
    title = doc.get("title", "Unknown")
    
    # 메인 노드 스타일
    safe_doc_id = doc_id.replace("-", "_")
    lines.append(f'    Main["{doc_id}<br/>({title})"]:::mainNode')
    
    # 참조하는 문서들 (Out-degree)
    for ref in refs.get("references", []):
        ref_id = ref.replace("-", "_")
        lines.append(f'    Main --> {ref_id}["{ref}"]')
        
    # 참조되는 문서들 (In-degree)
    for cited in refs.get("cited_by", []):
        cited_id = cited.replace("-", "_")
        lines.append(f'    {cited_id}["{cited}"] --> Main')
        
    lines.append("    classDef mainNode fill:#f96,stroke:#333,stroke-width:4px;")
    return "\n".join(lines)

def graph_agent_node(state: AgentState):
    """[서브] 그래프 에이전트 (Z.AI) - 인텐트 분석 및 시각화 지원"""
    client = get_zai_client()
    query = state["query"]
    model = state.get("worker_model") or state.get("model_name") or "glm-4.7-flash"
    
    # 1. 의도 및 엔티티 추출 (Z.AI 활용 - 대화 이력 포함)
    messages = state.get("messages", [])
    extraction_prompt = f"""사용자의 질문과 대화 이력을 분석하여 분석 대상이 되는 SOP ID와 질문의 의도를 추출하세요.
    - 이전 대화에서 언급된 문서 ID가 있다면 그것을 사용하세요.
    - 질문: {query}
    
    [의도 분류]
    - impact_analysis: 특정 문서를 변경했을 때 영향을 받는 하위 문서나 관련 절차를 찾고자 할 때
    - dependency_analysis: 특정 문서가 작동하기 위해 참조해야 하는 상위 규정이나 근거를 찾고자 할 때
    - relationship_check: 두 문서 사이의 연결 고리를 확인하고자 할 때
    - general_info: 단순히 특정 문서의 참조 목록을 보고 싶어할 때
    
    반드시 JSON 형식으로만 답변하세요.
    예: {{"doc_id": "EQ-SOP-001", "intent": "impact_analysis", "reason": "이전 대화에서 찾은 SOP-001의 영향 분석"}}"""
    
    try:
        extraction_res = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": "당신은 대화 맥락을 파석하여 엔티티를 추출하는 전문가입니다."}] + messages + [{"role": "user", "content": extraction_prompt}],
            response_format={"type": "json_object"}
        )
        # safe_json_loads를 통한 강인한 파싱
        info = safe_json_loads(extraction_res.choices[0].message.content)
        doc_id = normalize_doc_id(info.get("doc_id")) # 정규화 적용
        intent = info.get("intent", "general_info")
    except:
        # 추출 실패 시 정규식 보조
        raw_match = re.search(r'([A-Za-z0-9_-]*SOP-\d+)', query.upper())
        doc_id = normalize_doc_id(raw_match.group(1)) if raw_match else None
        intent = "general_info"
    
    if not doc_id:
        # SOP-로 시작하지 않는 문서명인 경우 다시 한번 시도
        raw_match = re.search(r'([A-Za-z0-9_-]+SOP[A-Za-z0-9_-]+)', query.upper())
        doc_id = normalize_doc_id(raw_match.group(1)) if raw_match else None
        
    if not doc_id:
        return {"messages": [{"role": "assistant", "content": "[그래프 에이전트] 분석할 문서 ID를 찾지 못했습니다. (예: SOP-001 관계 분석해줘)"}]}

    # 2. 데이터 조회 (Tool 활용)
    refs_str = get_references_tool.invoke({"doc_id": doc_id})
    
    if not refs_str or refs_str == "None":
        return {"messages": [{"role": "assistant", "content": f"[그래프 에이전트] {doc_id}에 대한 참조 데이터가 존재하지 않습니다."}]}

    # safe_json_loads를 통한 강인한 파싱
    ref_data = safe_json_loads(refs_str)
    if not ref_data:
        ref_data = {"document": {"doc_id": doc_id}, "references": [], "cited_by": []}

    # 3. 시각화 (Mermaid) 생성
    mermaid_code = generate_mermaid_flow(doc_id, ref_data)
    
    # 4. 심층 분석 (Z.AI)
    analysis_prompt = f"""다음 그래프 데이터를 바탕으로 질문에 대해 전문적인 분석 보고서를 작성하세요.
    질문: {query}
    의도: {intent}
    데이터: {json.dumps(ref_data, ensure_ascii=False)}
    
    [보고서 규칙]
    - **STRICT GROUNDING**: 오직 제공된 `데이터` 내의 관계 정보만 분석하세요.
    - **NO INFERENCE**: 문서 간의 관계가 데이터에 명시되지 않았다면 "관계를 알 수 없음"으로 보고하세요.
    - 질문의 의도({intent})에 맞춰 '영향'이나 '의존성'을 명확히 설명하세요.
    - '참조하는 문서(References)'는 상위 규정 또는 필수 참고서입니다.
    - '참조받는 문서(Cited By)'는 이 문서가 변경될 때 함께 업데이트되어야 할 대상입니다.
    - 전문적인 용어를 사용하되 불렛 포인트로 간결하게 정리하세요.
    - 한국어로 답변하세요.
    - **EXTERNAL KNOWLEDGE PROHIBITED**: 일반적인 지식이나 상식에 기반한 문서 간 관계 추측을 엄격히 금지합니다.
    """
    
    analysis_res = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": analysis_prompt}]
    )
    
    final_report = f"""###  {doc_id} 관계망 분석 보고서

{analysis_res.choices[0].message.content}

####  시각화 관계도 (Mermaid)
```mermaid
{mermaid_code}
```
"""
    
    # [중간 보고] 답변 에이전트가 사용할 수 있도록 context에 시각화 코드와 분석 내용을 함께 저장 (리스트 누적)
    report = f"### [그래프 에이전트 관계 분석 보고]\n{final_report}"
    return {"context": [report]}
