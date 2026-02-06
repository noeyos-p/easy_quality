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

def _normalize_plan(plan: List[str]) -> List[str]:
    """
    플랜 항목을 정규화하여 최상위 조항(Depth1)만 남김.
    예: ["1.1", "2.3.4", "목적5"] -> ["1", "2", "5"]
    """
    normalized = []
    for item in plan:
        # 숫자만 추출 (예: "1.1" -> "1", "5조" -> "5")
        match = re.search(r'^(\d+)', str(item).strip())
        if match:
            num = match.group(1)
            if num not in normalized:
                normalized.append(num)
    
    # 최대 15개로 제한
    return normalized[:15]

def planner_node(state: SummaryState):
    """[Planner] 질문 의도와 문서 구조를 파악하여 요약 계획 수립"""
    client = get_zai_client()
    headers_tool = get_headers_tool()
    query = state["query"]

    # 1. 문서 ID 추출
    id_prompt = f"다음 질문에서 분석 대상이 되는 문서 ID(예: EQ-SOP-00001)만 추출하세요. 질문: {query}"
    try:
        id_res = client.chat.completions.create(model=state["model"], messages=[{"role": "user", "content": id_prompt}])
        doc_id_match = re.search(r'([A-Z]{2}-SOP-\d+)', id_res.choices[0].message.content.upper())
        doc_id = doc_id_match.group(1) if doc_id_match else None
    except Exception as e:
        print(f"    [Deep Summary] 문서 ID 추출 실패: {e}")
        doc_id = None

    actual_headers = ""
    if doc_id:
        try:
            actual_headers = headers_tool.invoke({"doc_id": doc_id})
            print(f"    [Deep Summary] 실제 목차 파악 성공: {doc_id}")
        except Exception as e:
            print(f"    [Deep Summary] 목차 조회 실패: {e}")

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
        mode = decision.get("mode", "global")
        plan = decision.get("plan", [])
        
        # 플랜 정규화 (1.1 -> 1 승격 등)
        normalized_plan = _normalize_plan(plan)
        
        # 검증 및 폴백
        if mode == "section" and not normalized_plan:
            print("    [Deep Summary] 빈 플랜 발생, 실제 목차에서 폴백 추출 시작")
            # 보수적인 정규식: 숫자 뒤에 공백, 마침표, 또는 닫는 괄호가 오는 패턴
            # 예: "1. 목적", "2) 범위", "3 정의"
            fallback_matches = re.findall(r'^-?\s*(\d+)[\.\s\)]', actual_headers, re.MULTILINE)
            normalized_plan = sorted(list(set(fallback_matches)), key=int)[:15]
            
            if not normalized_plan:
                print("    [Deep Summary] 폴백 추출도 실패함. global 모드로 강등.")
                mode = "global"
        
        return {
            "doc_id": doc_id or decision.get("doc_id"),
            "summary_mode": mode,
            "plan": normalized_plan,
            "current_step": 0
        }
    except Exception as e:
        print(f"    [Deep Summary] 플래너 오류: {e}, global 모드로 강등")
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
    target_title = "" # 검색 품질을 위해 타이틀 확보 시도

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
        except Exception as e:
            print(f"    [Deep Summary] 그래프 DB 조회 실패: {e}, 단일 조항으로 진행")
            all_section_ids = [f"{doc_id}:{target_clause}"]

    # 2. 그래프 DB에서 직접 각 조항의 내용 가져오기
    all_results = []
    for section_id in all_section_ids:
        # section_id에서 조항 번호만 추출 (예: "EQ-SOP-00001:1.1" -> "1.1")
        clause_num = section_id.split(':')[-1]

        if graph_store:
            try:
                section_data = graph_store.get_section_content(section_id)
                if section_data:
                    title = section_data.get('title', '')
                    content = section_data.get('content', '')
                    
                    if clause_num == target_clause:
                        target_title = title

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
            except Exception as e:
                print(f"    [Deep Summary] 조항 조회 실패: {e}")

    # 3. 데이터가 부족하거나 그래프 DB에 없는 경우 검색 도구로 보강
    if not all_results or len(all_results) < 1:
        print(f"    [Deep Summary] 그래프 데이터 부족, 검색 보강 시도: {target_clause}")
        
        # [품질 개선] 질문에서 핵심 키워드 추출 (예: Legacy 등)
        # 조항 번호 단독 검색은 노이즈가 크므로 반드시 doc_id와 제목/키워드를 포함
        extra_keywords = ""
        # 질문 내 명사류나 영어 단어가 있으면 검색 품질 향상을 위해 쿼리에 포함
        potential_keywords = re.findall(r'[a-zA-Z가-힣]{2,}', query)
        if potential_keywords:
            # 질문에서 조항 번호나 일반적인 단어 제외하고 2~3개만 추출
            filtered = [k for k in potential_keywords if k not in ["SOP", "summary", "요약", "섹션", "조항"]]
            extra_keywords = " ".join(filtered[:3])

        search_query = f"{doc_id} {target_clause} {target_title} {extra_keywords}".strip()
        search_res = search_tool.invoke({
            "query": search_query,
            "target_doc_id": doc_id,
            "keywords": [target_clause]
        })
        if search_res and "검색 결과 없음" not in search_res:
            all_results.append(search_res)

    # 4. 모든 결과 병합
    if all_results:
        combined = "\n\n---\n\n".join(all_results)
    else:
        combined = "(본문에 명시 없음)"

    return {
        "full_context": [f"### [제{target_clause}조 {target_title} 통합 데이터]\n{combined}"],
        "current_step": step + 1
    }

def finalizer_node(state: SummaryState):
    """[Finalizer] 수집된 모든 정보를 취합하여 최종 답변 생성"""
    client = get_zai_client()
    query = state["query"]
    contexts = "\n\n".join(state["full_context"])
    mode = state["summary_mode"]
    
    if mode == "section":
        system_prompt = """당신은 SOP 전문 분석가입니다. 수집된 [데이터]에만 기반하여 **조항별 최종 요약 보고서**를 작성하세요.

**중요 출력 규격 및 규칙 (반드시 준수):**
1. **헤더 고정**: 헤더는 반드시 정수(1, 2, 3...) 형태의 Depth1 조항 번호만 사용하세요. (예: "1. 목적:", "2. 적용 범위:")
2. **하위 섹션 제목화 금지**: 1.1, 5.2.2 등의 하위 조항 번호를 **별도의 섹션 제목(헤더)으로 절대 쓰지 마세요.**
3. **내용 흡수**: 모든 하위 조항의 상세 내용은 해당 부모 정수 섹션(Depth1) 내부에 불릿 포인트(-) 형식으로만 기술하세요.
4. **근거 엄격성**: [데이터]에 해당 조항과 관련된 단 한 줄의 내용이라도 있다면 절대로 "(본문에 명시 없음)"이라고 하지 마세요. 반드시 그 내용을 요약에 반영해야 합니다.
5. **추측 금지**: [데이터]에 없는 내용은 절대 지어내거나 일반론(보통, 일반적으로 등)으로 채우지 마세요. 정말 데이터가 0인 경우에만 **"(본문에 명시 없음)"**으로 표기하세요.
6. **군더더기 제거**: 보고서 전후에 '총평', '요약', '결론' 등의 추가 섹션을 절대 만들지 마세요. 오직 정수 조항별 요약 리스트만 출력하세요."""
    else:
        system_prompt = """당신은 SOP 전문 분석가입니다. 수집된 [데이터]에만 기반하여 문서 전체의 핵심을 요약하세요.
- **[데이터]에 없는 내용은 절대 추측하거나 지어내지 마세요.**
- 5~8개의 핵심 문장으로 정리하세요.
- 한국어로 친절하게 답변하세요."""
        
    res = client.chat.completions.create(
        model=state["model"],
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"질문: {query}\n\n[데이터]\n{contexts}"}
        ],
        temperature=0.1
    )
    
    report_tag = "[딥 에이전트 - 조항별 상세 요약]" if mode == "section" else "[딥 에이전트 - 전체 핵심 요약]"
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
