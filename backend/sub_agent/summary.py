"""
문서 요약 서브에이전트 모듈 (Deep Agent 스타일)
- 질문 분석 후 계획을 세우고 조항별로 정밀하게 요약하는 그래프 구조의 에이전트
- 벡터 검색 (Weaviate), SQL 검색 (PostgreSQL), 그래프 검색 (Neo4j) 통합
"""

import json
import re
import operator
from typing import Any, Dict, List, Optional, Annotated, TypedDict, Literal

from langsmith import traceable
from langgraph.graph import StateGraph, START, END

# ═══════════════════════════════════════════════════════════════════════════
# 전역 스토어 및 클라이언트 관리
# ═══════════════════════════════════════════════════════════════════════════

_zai_client = None
_search_tool = None
_headers_tool = None
_graph_store = None

def init_summary_stores(graph_store_instance=None):
    """요약 에이전트용 스토어 및 도구 초기화"""
    global _search_tool, _headers_tool, _graph_store
    # 실제 도구는 agent.py에서 가져옴 (필요 시 lazy import)
    from backend.agent import search_sop_tool, get_sop_headers_tool
    _search_tool = search_sop_tool
    _headers_tool = get_sop_headers_tool
    _graph_store = graph_store_instance

def get_zai_client():
    """Z.AI 클라이언트 반환"""
    global _zai_client
    if not _zai_client:
        from backend.agent import get_zai_client as get_main_zai
        _zai_client = get_main_zai()
    return _zai_client

def get_search_tool():
    """검색 도구 반환"""
    global _search_tool
    if not _search_tool:
        init_summary_stores()
    return _search_tool

def get_headers_tool():
    """헤더 조회 도구 반환"""
    global _headers_tool
    if not _headers_tool:
        init_summary_stores()
    return _headers_tool

def get_graph_store():
    """그래프 스토어 반환"""
    global _graph_store
    return _graph_store

# ═══════════════════════════════════════════════════════════════════════════
# 딥 에이전트 상태 정의 (SummaryState)
# ═══════════════════════════════════════════════════════════════════════════

class SummaryState(TypedDict):
    """
    요약 에이전트 상태.
    'messages' 키를 포함하여 대화 이력을 관리합니다.
    """
    messages: Annotated[List[Any], operator.add]
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
    headers_tool = get_headers_tool()
    query = state["query"]

    # 1. 문서 ID 추출 및 실제 목차 조회
    id_prompt = f"다음 질문에서 분석 대상이 되는 문서 ID(예: EQ-SOP-00001)만 추출하세요. 질문: {query}"
    id_res = client.chat.completions.create(model=state["model"], messages=[{"role": "user", "content": id_prompt}])
    doc_id = re.search(r'([A-Z]{2}-SOP-\d+)', id_res.choices[0].message.content.upper())
    doc_id = doc_id.group(1) if doc_id else None

    actual_headers = ""
    if doc_id:
        actual_headers = headers_tool.invoke({"doc_id": doc_id})
        print(f"    [Deep Summary] 실제 목차 파악 성공: {doc_id}")

    # 2. 요약 모드 결정 및 계획 수립
    prompt = f"""사용자의 질문을 분석하여 요약 계획을 세우세요.
    질문: {query}
    문서 ID: {doc_id}
    실제 조항 목록:
    {actual_headers}

    [작업]
    1. 요약 모드 결정 (global: 전체 핵심, section: 조항별 상세)
    2. section 모드인 경우, 최상위 조항(depth 0)만 선택하세요.
       - 예: "1", "2", "3" (O) / "1.1", "2.3.4" (X)
       - 최상위 조항 하나가 선택되면 그 하위 조항들(1.1, 1.2 등)은 자동으로 포함됩니다.
    3. 발견된 '실제 조항 목록' 중 질문과 관련이 있거나 요약해야 할 최상위 조항 번호들을 선택하세요.
    4. **절대 조항 번호를 지어내지 말고, 위의 목록에 있는 번호만 사용하세요.**

    반드시 JSON으로 답변하세요:
    {{"doc_id": "{doc_id}", "mode": "global|section", "plan": ["1", "2", "5"]}}"""
    
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
    """[Worker] 계획된 조항별로 정밀 검색 수행
    - 그래프 DB를 활용하여 최상위 조항의 모든 하위 조항을 재귀적으로 조회
    - 예: "1"을 선택하면 Graph DB에서 PARENT_OF 관계를 따라 1.1, 1.2, 1.2.1 등을 조회
    """
    search_tool = get_search_tool()
    graph_store = get_graph_store()
    query = state["query"]
    doc_id = state["doc_id"]
    plan = state["plan"]
    step = state["current_step"]

    # 계획이 없거나 global 모드면 일반 검색
    if not plan or state["summary_mode"] == "global":
        search_res = search_tool.invoke({
            "query": f"{doc_id} {query}",
            "target_doc_id": doc_id # 특정 문서로 한정
        })
        return {"full_context": [search_res], "current_step": step + 1}

    # 최상위 조항별 검색 (그래프 DB로 하위 조항 조회)
    target_clause = plan[step]

    # 1. 그래프 DB에서 하위 조항 리스트 가져오기
    all_section_ids = []  # 전체 section_id 형식 (EQ-SOP-00001:1.1)
    if graph_store:
        try:
            # section_id 형식: "EQ-SOP-00001:1"
            full_section_id = f"{doc_id}:{target_clause}"
            subsections = graph_store.get_subsections_recursive(doc_id, full_section_id)

            if subsections:
                all_section_ids = subsections
                # section_id에서 조항 번호만 추출 (예: "EQ-SOP-00001:1.1" -> "1.1")
                clause_numbers = [s.split(':')[-1] for s in subsections]
                print(f"    [Deep Summary] {doc_id} 제{target_clause}조 하위 조항 발견: {clause_numbers}")
            else:
                # 하위 조항이 없으면 자기 자신만
                all_section_ids = [full_section_id]
                print(f"    [Deep Summary] {doc_id} 제{target_clause}조 하위 조항 없음 (단독 조항)")
        except Exception as e:
            print(f"    [Deep Summary] 그래프 DB 조회 실패: {e}, 단일 조항으로 진행")
            all_section_ids = [f"{doc_id}:{target_clause}"]

    # 2. 그래프 DB에서 직접 각 조항의 내용 가져오기
    all_results = []
    for section_id in all_section_ids:
        # section_id에서 조항 번호만 추출 (예: "EQ-SOP-00001:1.1" -> "1.1")
        clause_num = section_id.split(':')[-1]
        print(f"    [Deep Summary] {doc_id} {clause_num}조 본문 가져오는 중...")

        if graph_store:
            try:
                section_data = graph_store.get_section_content(section_id)
                if section_data:
                    title = section_data.get('title', '')
                    content = section_data.get('content', '')

                    # 제목과 내용 조합
                    if title and content:
                        section_text = f"**{clause_num}조: {title}**\n\n{content}"
                    elif title:
                        section_text = f"**{clause_num}조: {title}**"
                    elif content:
                        section_text = f"**{clause_num}조**\n\n{content}"
                    else:
                        section_text = f"**{clause_num}조** (내용 없음)"

                    all_results.append(section_text)
                else:
                    print(f"    [Deep Summary] {section_id} 조항 데이터 없음")
            except Exception as e:
                print(f"    [Deep Summary] 조항 조회 실패: {e}")

    # 3. 모든 결과 병합
    if all_results:
        combined = "\n\n---\n\n".join(all_results)
    else:
        combined = f"제{target_clause}조에 대한 데이터가 없습니다."

    return {
        "full_context": [f"### [제{target_clause}조 및 하위 조항 통합 데이터]\n{combined}"],
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
        - 각 최상위 조항(1조, 2조 등)별로 핵심 내용을 종합하여 정리하세요.
        - 각 조항 내의 하위 항목들(1.1, 1.2 등)은 이미 데이터에 포함되어 있으므로, 이를 통합하여 해당 조항의 전체적인 내용을 요약하세요.
        - 불릿 포인트로 간결하게 정리하되, 조항의 목적과 주요 내용이 드러나도록 작성하세요.
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
    
    report_tag = "[딥 에이전트 - 주요 조항별 통합 요약]" if mode == "section" else "[딥 에이전트 - 전체 핵심 요약]"
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

@traceable(name="sub_agent:summary")
def summary_agent_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """[서브] 요약 에이전트 (Deep Agent 버전)
    - 내부 그래프를 통해 스스로 계획을 세우고 조항별로 정밀하게 읽습니다.
    """
    global _deep_summary_app
    if not _deep_summary_app:
        _deep_summary_app = create_deep_summary_graph()

    print(f" [Deep Summary] 딥 에이전트 가동 시작: {state['query']}")

    initial_summary_state = {
        "messages": [{"role": "user", "content": state["query"]}],
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
