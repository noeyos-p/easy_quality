import json
import re
import operator
from typing import Any, Dict, List, Optional, Annotated, TypedDict, Literal
from backend.agent import get_zai_client, search_sop_tool, get_sop_headers_tool, AgentState
from langgraph.graph import StateGraph, START, END

# ═══════════════════════════════════════════════════════════════════════════
# 딥 에이전트 상태 정의 (SummaryState)
# ═══════════════════════════════════════════════════════════════════════════

class SummaryState(TypedDict):
    query: str
    doc_id: Optional[str]
    full_context: Annotated[List[str], operator.add]
    summary_mode: Literal["global", "section"]
    plan: List[str] # 요약할 조항/섹션 리스트
    current_step: int
    final_report: str
    model: str

# ═══════════════════════════════════════════════════════════════════════════
# 노드 정의 (Nodes)
# ═══════════════════════════════════════════════════════════════════════════

def planner_node(state: SummaryState):
    """[Planner] 질문 의도와 문서 구조를 파악하여 요약 계획 수립"""
    client = get_zai_client()
    query = state["query"]
    
    # 1. 문서 ID 추출 및 실제 목차 조회
    id_prompt = f"다음 질문에서 분석 대상이 되는 문서 ID(예: EQ-SOP-00001)만 추출하세요. 질문: {query}"
    id_res = client.chat.completions.create(model=state["model"], messages=[{"role": "user", "content": id_prompt}])
    doc_id = re.search(r'([A-Z]{2}-SOP-\d+)', id_res.choices[0].message.content.upper())
    doc_id = doc_id.group(1) if doc_id else None
    
    actual_headers = ""
    if doc_id:
        actual_headers = get_sop_headers_tool.invoke({"sop_id": doc_id})
        print(f"   📑 [Deep Summary] 실제 목차 파악 성공: {doc_id}")

    # 2. 요약 모드 결정 및 계획 수립
    prompt = f"""사용자의 질문을 분석하여 요약 계획을 세우세요.
    질문: {query}
    문서 ID: {doc_id}
    실제 조항 목록:
    {actual_headers}
    
    [작업]
    1. 요약 모드 결정 (global: 전체 핵심, section: 조항별 상세)
    2. 발견된 '실제 조항 목록' 중 질문과 관련이 있거나 요약해야 할 조항 번호들을 선택하세요.
    3. **절대 조항 번호를 지어내지 말고, 위의 목록에 있는 번호만 사용하세요.**
    
    반드시 JSON으로 답변하세요: 
    {{"doc_id": "{doc_id}", "mode": "global|section", "plan": ["1.1", "2.1", "5.4"]}}"""
    
    try:
        res = client.chat.completions.create(
            model=state["model"],
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        decision = json.loads(res.choices[0].message.content)
        return {
            "doc_id": decision.get("doc_id"),
            "summary_mode": decision.get("mode", "global"),
            "plan": decision.get("plan", []),
            "current_step": 0
        }
    except:
        return {"summary_mode": "global", "plan": [], "current_step": 0}

def worker_node(state: SummaryState):
    """[Worker] 계획된 조항별로 정밀 검색 수행 (캐싱 효과)"""
    query = state["query"]
    doc_id = state["doc_id"]
    plan = state["plan"]
    step = state["current_step"]
    
    # 계획이 없거나 global 모드면 일반 검색
    if not plan or state["summary_mode"] == "global":
        search_res = search_sop_tool.invoke({
            "query": f"{doc_id} {query}",
            "target_sop_id": doc_id # 특정 문서로 한정
        })
        return {"full_context": [search_res], "current_step": step + 1}
    
    # 조항별 검색 (현재 스텝의 조항) - 정밀 타격
    target_clause = plan[step]
    print(f"   🔍 [Deep Summary] {doc_id} {target_clause}조 본문 타격 중...")
    
    search_query = f"{target_clause}"
    search_res = search_sop_tool.invoke({
        "query": search_query, 
        "target_sop_id": doc_id, # 다른 문서 노이즈 차단
        "keywords": [target_clause]
    })
    
    return {
        "full_context": [f"### [제{target_clause}조 실제 본문 데이터]\n{search_res}"],
        "current_step": step + 1
    }

def finalizer_node(state: SummaryState):
    """[Finalizer] 수집된 모든 정보를 취합하여 최종 답변 생성"""
    client = get_zai_client()
    query = state["query"]
    contexts = "\n\n".join(state["full_context"])
    mode = state["summary_mode"]
    
    if mode == "section":
        system_prompt = """당신은 SOP 전문 분석가입니다. 수집된 조항별 데이터를 바탕으로 **사용자에게 바로 전달할 최종 보고서**를 작성하세요.
        - 각 조항별로 핵심 내용을 불릿 포인트로 정리하세요.
        - 누락된 조항이 있다면 아는 범위 내에서 정리하되, 가급적 수집된 데이터에 충실하세요.
        - 한국어로 친절하게 답변하세요."""
    else:
        system_prompt = """당신은 SOP 전문 분석가입니다. 수집된 데이터를 바탕으로 문서 전체의 핵심을 요약하여 **최종 답변**을 작성하세요.
        - 5~8개의 핵심 문장으로 정리하세요.
        - 한국어로 친절하게 답변하세요."""
        
    res = client.chat.completions.create(
        model=state["model"],
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"질문: {query}\n\n[수집된 데이터]\n{contexts}"}
        ]
    )
    
    report_tag = "[딥 에이전트 - 조항별 정밀 분석]" if mode == "section" else "[딥 에이전트 - 전체 핵심 요약]"
    return {"final_report": f"{report_tag}\n{res.choices[0].message.content}\n\n[DONE]"}

# ═══════════════════════════════════════════════════════════════════════════
# 그래프 구성
# ═══════════════════════════════════════════════════════════════════════════

def create_deep_summary_graph():
    workflow = StateGraph(SummaryState)
    
    workflow.add_node("planner", planner_node)
    workflow.add_node("worker", worker_node)
    workflow.add_node("finalizer", finalizer_node)
    
    workflow.add_edge(START, "planner")
    workflow.add_edge("planner", "worker")
    
    def should_continue(state: SummaryState):
        # 계획된 모든 조항을 다 읽었거나, global 모드면 종료 단계로
        if state["summary_mode"] == "global" or state["current_step"] >= len(state["plan"]):
            return "finalizer"
        # 더 읽어야 할 조항이 남았다면 Worker 반복
        return "worker"
    
    workflow.add_conditional_edges(
        "worker",
        should_continue,
        {
            "worker": "worker",
            "finalizer": "finalizer"
        }
    )
    
    workflow.add_edge("finalizer", END)
    return workflow.compile()

# ═══════════════════════════════════════════════════════════════════════════
# 메인 엔트리 포인트 (외부에서 호출되는 함수)
# ═══════════════════════════════════════════════════════════════════════════

_deep_summary_app = None

def summary_agent_node(state: AgentState):
    """[서부] 요약 에이전트 (Deep Agent 버전)
    - 내부 그래프를 통해 스스로 계획을 세우고 조항별로 정밀하게 읽습니다.
    """
    global _deep_summary_app
    if not _deep_summary_app:
        _deep_summary_app = create_deep_summary_graph()
        
    print(f"🚀 [Deep Summary] 딥 에이전트 가동 시작: {state['query']}")
    
    initial_summary_state = {
        "query": state["query"],
        "doc_id": None,
        "full_context": [],
        "summary_mode": "global",
        "plan": [],
        "current_step": 0,
        "model": state.get("worker_model") or state.get("model_name") or "glm-4.7-flash",
        "final_report": ""
    }
    
    # 내부 딥 루프 실행 (최대 15단계 제한)
    result = _deep_summary_app.invoke(initial_summary_state, config={"recursion_limit": 15})
    
    return {"messages": [{"role": "assistant", "content": result["final_report"]}]}
