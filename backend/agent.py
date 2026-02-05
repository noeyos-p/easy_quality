"""
SOP 멀티 에이전트 시스템 v13.0
- Orchestrator (Main): OpenAI (GPT-4o-mini) - 질문 분석 및 라우팅, 최종 답변
- Specialized Sub-Agents: Z.AI (GLM-4.7) - 실행 및 데이터 처리
  1. Retrieval Agent: 문서 검색 및 추출
  2. Summarization Agent: 문서/조항 요약
  3. Comparison Agent: 버전 비교
  4. Graph Agent: 참조 관계 조회
"""

import os
import re
import json
import operator
import hashlib
from typing import List, Dict, Optional, Any, Annotated, TypedDict, Literal
from datetime import datetime

# ═══════════════════════════════════════════════════════════════════════════

# 임포트 및 설정
# ═══════════════════════════════════════════════════════════════════════════

try:
    from openai import OpenAI
except ImportError:
    pass

try:
    from zai import ZaiClient
    ZAI_AVAILABLE = True
except ImportError:
    ZAI_AVAILABLE = False
    pass

try:
    from langchain_core.tools import tool
    from langgraph.graph import StateGraph, START, END
    from langsmith import traceable
    LANGCHAIN_AVAILABLE = True
    LANGGRAPH_AGENT_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    LANGGRAPH_AGENT_AVAILABLE = False
    pass

# ═══════════════════════════════════════════════════════════════════════════
# 전역 스토어 및 클라이언트
# ═══════════════════════════════════════════════════════════════════════════

_vector_store = None
_graph_store = None
_sql_store = None

_openai_client = None
_zai_client = None

def init_agent_tools(vector_store_module, graph_store_instance, sql_store_instance=None):
    global _vector_store, _graph_store, _sql_store
    _vector_store = vector_store_module
    _graph_store = graph_store_instance
    _sql_store = sql_store_instance

def get_openai_client():
    global _openai_client
    if not _openai_client:
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            _openai_client = OpenAI(api_key=api_key)
    return _openai_client

def get_zai_client():
    global _zai_client
    if not _zai_client:
        api_key = os.getenv("ZAI_API_KEY")
        if api_key:
            _zai_client = ZaiClient(api_key=api_key)
    return _zai_client

# ═══════════════════════════════════════════════════════════════════════════
# 도구 정의 (Tools)
# ═══════════════════════════════════════════════════════════════════════════

@tool
def search_sop_tool(query: str, extract_english: bool = False, keywords: List[str] = None) -> str:
    """SOP 문서 검색 도구.
    Hybrid Search(BM25 + Vector) 방식을 사용하여 키워드와 의미론적 연관성을 동시에 고려합니다.
    extract_english: True면 영문 내용 위주로 추출
    """
    global _vector_store, _sql_store
    
    results = []
    seen_content = set() # 중복 내용 방지
    
    # 1. 벡터 스토어의 하이브리드 검색 활용 (v8.0+)
    if _vector_store:
        search_query = query
        if keywords:
            # 키워드가 있으면 쿼리에 보강하여 BM25 점수 가중치 부여
            search_query += " " + " ".join(keywords)
            
        # 하이브리드 검색 수행 (alpha=0.4: 키워드 비중 약간 높임)
        vec_res = []
        try:
            # vector_store 모듈에 구현된 search_hybrid 호출
            vec_res = _vector_store.search_hybrid(search_query, n_results=10, alpha=0.4)
        except AttributeError:
            # 만약 구현이 아직 안되었다면 기본 search 사용
            vec_res = _vector_store.search(search_query, n_results=10)
            
        for r in vec_res:
            meta = r.get('metadata', {})
            sop_id = meta.get('sop_id') or meta.get('doc_name', 'Unknown')
            section = meta.get('section') or meta.get('clause') or "본문"
            content = r.get('text', '')
            
            if not content: continue
            
            # 해시로 중복 체크
            content_hash = hashlib.md5(content.encode()).hexdigest()
            if content_hash in seen_content: continue
            seen_content.add(content_hash)
            
            display_header = f"📄 [검색] {sop_id} > {section}"
            
            if extract_english:
                # 영문 추출 로직: 알파벳 비율이 한글보다 높은 문단 필터링
                paragraphs = content.split('\n\n')
                eng_paras = []
                for p in paragraphs:
                    eng_count = len(re.findall(r'[a-zA-Z]', p))
                    kor_count = len(re.findall(r'[가-힣]', p))
                    if eng_count > kor_count and eng_count > 10:
                        eng_paras.append(p)
                
                if eng_paras:
                    results.append(f"{display_header} (영문):\n" + "\n\n".join(eng_paras[:3]))
                else:
                    results.append(f"{display_header}:\n{content[:1500]}...")
            else:
                results.append(f"{display_header}:\n{content}")

    # 2. 결과가 전혀 없거나 매우 적을 경우 SQL 키워드 매칭 (보조/확정적 검색)
    if len(results) < 2 and _sql_store and keywords:
        all_docs = _sql_store.list_documents()
        for doc in all_docs:
            doc_name = doc.get('doc_name', '')
            # 문서명에 키워드가 포함된 경우
            if any(k.upper() in doc_name.upper() for k in keywords):
                doc_id = doc.get('id')
                sop_doc = _sql_store.get_document_by_id(doc_id)
                if sop_doc:
                    full_content = sop_doc.get("content", "")
                    if full_content:
                        results.append(f"📄 [문서 전체 가이드] {doc_name}:\n{full_content[:2000]}...")
                
    return "\n\n".join(results) if results else "검색 결과 없음. 검색어나 키워드를 바꿔보세요."

@tool
def get_version_history_tool(sop_id: str) -> str:
    """특정 문서의 버전 히스토리를 조회"""
    global _sql_store
    if not _sql_store: return "SQL 저장소 연결 실패"
    
    versions = _sql_store.get_document_versions(sop_id)
    if not versions: return f"{sop_id} 문서의 버전을 찾을 수 없습니다."
    
    return "\n".join([f"- v{v['doc_metadata'].get('version')} ({v['created_at']})" for v in versions])

@tool
def compare_versions_tool(sop_id: str, v1: str, v2: str) -> str:
    """두 버전의 문서 내용을 비교하여 반환"""
    global _sql_store
    if not _sql_store: return ""
    
    doc1 = _sql_store.get_document_by_id(sop_id, v1)
    doc2 = _sql_store.get_document_by_id(sop_id, v2)
    
    if not doc1 or not doc2: return "비교할 버전을 찾을 수 없습니다."
    
    return f"=== v{v1} ===\n{doc1.get('markdown_content')[:2000]}\n\n=== v{v2} ===\n{doc2.get('markdown_content')[:2000]}"

@tool
def get_references_tool(sop_id: str) -> str:
    """참조 관계 조회"""
    global _graph_store
    if not _graph_store: return ""
    refs = _graph_store.get_document_references(sop_id)
    return str(refs)

# ═══════════════════════════════════════════════════════════════════════════
# Agent State
# ═══════════════════════════════════════════════════════════════════════════

class AgentState(TypedDict):
    query: str
    messages: Annotated[List[Any], operator.add]
    next_agent: Literal["retrieval", "summary", "comparison", "graph", "end"]
    final_answer: str
    context: str
    model_name: Optional[str] # (레거시 호환용)
    worker_model: Optional[str] # 서브 에이전트(Worker)용 모델
    orchestrator_model: Optional[str] # 오케스트레이터용 모델

# ═══════════════════════════════════════════════════════════════════════════
# 노드 정의 (Nodes)
# ═══════════════════════════════════════════════════════════════════════════

def orchestrator_node(state: AgentState):
    """
    메인 에이전트 (OpenAI GPT-4o-mini)
    - 사용자의 질문을 분석하여 적절한 서브 에이전트로 라우팅하거나 최종 답변을 생성
    """
    client = get_openai_client()
    messages = state["messages"]
    
    system_prompt = """당신은 GMP 규정 시스템의 메인 오케스트레이터(Manager)입니다.
    사용자의 질문을 해결하기 위해 하위 전문가 에이전트들을 지휘하고, 그들의 보고를 검증하는 역할을 수행합니다.
    
    [작업 흐름]
    1. **History 분석**: 이전 대화 내용(History)을 보고, 이미 수행된 에이전트의 보고가 있는지 확인하세요.
    2. **판단(Judgement)**: 
       - 보고 내용이 충분하다면 -> 'finish'를 선택하여 최종 답변을 작성하세요.
       - 보고 내용이 부족하거나 오류가 있다면 -> 다른 에이전트를 호출하거나, 검색 조건을 바꿔서 다시 시도하게 하세요.
       - 아직 시작 단계라면 -> 적절한 에이전트를 호출하세요.
    
    [에이전트 목록 및 라우팅 가이드]
    1. retrieval: 규정 검색, 정보 조회. (어떤 문서가 있는지 모를 때 먼저 사용)
    2. summary: 문서나 조항의 요약. (이미 찾은 문서의 내용을 요약할 때 사용)
    3. comparison: 두 문서의 버전 차이 비교.
    4. graph: **참조/인용 관계(Reference), 상위/하위 규정 관계 확인**. "참조 목록 알려줘", "어떤 규정을 따르나?", "영향 분석해줘" 등의 질문은 반드시 이 에이전트가 처리해야 합니다.
    
    [라우팅 규칙]
    - 사용자가 "참조 목록", "Reference", "연결된 문서" 등을 물어보면 **무조건 `graph`**를 호출하세요. `retrieval`로 본문에서 찾으려 하지 마세요.
    - 이전 대화에서 이미 특정 문서(SOP ID)가 식별되었다면, 그 ID를 바탕으로 전문 에이전트(summary, graph)를 즉시 호출하세요.
    - 만약 서브 에이전트가 "문서 ID를 찾지 못했다"고 보고한다면, `retrieval`을 통해 먼저 문서 ID를 찾은 후 다시 해당 에이전트를 부르세요.
    
    [출력 형식]
    JSON 형식으로 'next_action' (agent 이름 또는 'finish')과 'reason'을 반환하세요.
    - **중요(Termination)**: 서브 에이전트의 보고 내용에 이미 답변에 필요한 충분한 정보(예: 검색된 문단, 시각화 보고서, 요약 등)가 있다면 즉시 'finish'를 선택하세요.
    - **루프 방지(Loop Prevention)**: 
        1. 동일한 서브 에이전트({next_action})를 같은 목적({reason})으로 3회 이상 반복 호출하지 마세요. 
        2. 만약 `retrieval` 에이전트가 "검색 결과 없음"을 보고했다면, 똑같은 검색어로는 다시 호출하지 마세요. 검색어를 바꾸거나 실패를 인정하고 'finish'하세요.
        3. 이미 답변할 근거가 생겼음에도 불구하고 서브 에이전트를 계속 부르는 것은 금지됩니다.
    
    예: {"next_action": "retrieval", "reason": "규정 검색 결과가 부족하여 재검색 필요"}
    """
    
    # 마지막 메시지가 도구 결과(Context)라면 답변 생성 모드로 진입 확률 높음
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": system_prompt}] + messages,
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        content = response.choices[0].message.content
        decision = json.loads(content)
        
        next_action = decision.get("next_action", "finish")
        
        # 만약 finish라면 최종 답변 생성
        if next_action == "finish":
            # 한 번 더 호출하여 자연어 답변 생성
            final_res = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": "수집된 정보를 바탕으로 사용자에게 친절하게 최종 답변을 하세요. 한국어로 답변해."}] + messages
            )
            return {"next_agent": "end", "final_answer": final_res.choices[0].message.content}
            
        return {"next_agent": next_action}
        
    except Exception as e:
        print(f"Orchestrator Error: {e}")
        return {"next_agent": "end", "final_answer": "오류가 발생했습니다."}

def retrieval_agent_node(state: AgentState):
    """[서브] 검색 에이전트 (Z.AI)"""
    client = get_zai_client()
    query = state["query"]
    
    # 한글/영문 요청 분석
    is_english_req = "영문" in query or "영어" in query or "english" in query.lower()
    
    # 1. 도구 실행 (직접 호출)
    # 실제로는 LLM이 도구 인자를 결정하게 할 수 있으나 여기선 규칙 기반으로 단순화
    search_res = search_sop_tool.invoke({"query": query, "extract_english": is_english_req, "keywords": query.split()})
    
    # 2. 결과 정리 (Z.AI)
    prompt = f"""다음 검색 결과를 바탕으로 사용자 질문에 답하세요.
    질문: {query}
    
    [검색 결과]
    {search_res}
    
    요약해서 핵심만 전달하세요."""
    
    res = client.chat.completions.create(
        model=state.get("worker_model") or state.get("model_name") or "glm-4.7-flash", # 동적 모델 적용
        messages=[{"role": "user", "content": prompt}]
    )
    
    return {"messages": [{"role": "assistant", "content": f"[검색 에이전트 보고]\n{res.choices[0].message.content}"}]}

def comparison_agent_node(state: AgentState):

    """[서브] 비교 에이전트 (Z.AI)"""
    client = get_zai_client()
    query = state["query"]
    
    # ID 및 버전 추출 로직은 복잡하므로 간단히 가정
    # 예: "SOP-001 버전 1.0과 2.0 비교해줘" -> 정규식으로 추출 필요
    # 여기서는 데모용으로 하드코딩된 로직 대신 LLM에게 추출 유도 가능
    
    res = client.chat.completions.create(
        model=state.get("worker_model") or state.get("model_name") or "glm-4.7-flash", # 동적 모델 적용
        messages=[{"role": "user", "content": f"사용자 질문 '{query}'에서 문서 ID와 버전 두 개를 추출해서 JSON으로 줘. 형식: {{'id': '...', 'v1': '...', 'v2': '...'}} "}]
    )
    try:
        info = json.loads(res.choices[0].message.content)
        comp_res = compare_versions_tool.invoke({"sop_id": info['id'], "v1": info['v1'], "v2": info['v2']})
        
        final_res = client.chat.completions.create(
            model=state.get("worker_model") or state.get("model_name") or "glm-4.7-flash", # 동적 모델 적용
            messages=[{"role": "user", "content": f"두 버전의 차이점을 분석해줘:\n{comp_res}"}]
        )
        content = final_res.choices[0].message.content
    except:
        content = "버전 정보를 정확히 추출하지 못했습니다. (예: SOP-001 1.0과 2.0 비교해줘)"

    return {"messages": [{"role": "assistant", "content": f"[비교 에이전트 보고]\n{content}"}]}



# ═══════════════════════════════════════════════════════════════════════════
# 워크플로우 구성
# ═══════════════════════════════════════════════════════════════════════════

def create_workflow():
    from backend.sub_agent.summary import summary_agent_node
    from backend.sub_agent.graph import graph_agent_node
    workflow = StateGraph(AgentState)

    
    # Nodes
    workflow.add_node("orchestrator", orchestrator_node)
    workflow.add_node("retrieval", retrieval_agent_node)
    workflow.add_node("summary", summary_agent_node)
    workflow.add_node("comparison", comparison_agent_node)
    workflow.add_node("graph", graph_agent_node)
    
    # Edges
    workflow.add_edge(START, "orchestrator")
    
    # Router
    def router(state: AgentState):
        return state["next_agent"]
    
    workflow.add_conditional_edges(
        "orchestrator",
        router,
        {
            "retrieval": "retrieval",
            "summary": "summary",
            "comparison": "comparison",
            "graph": "graph",
            "end": END
        }
    )
    
    # 각 서브 에이전트는 다시 오케스트레이터로 돌아와서 결과를 보고함
    workflow.add_edge("retrieval", "orchestrator")
    workflow.add_edge("summary", "orchestrator")
    workflow.add_edge("comparison", "orchestrator")
    workflow.add_edge("graph", "orchestrator")
    
    return workflow.compile()

# ═══════════════════════════════════════════════════════════════════════════
# 실행 인터페이스
# ═══════════════════════════════════════════════════════════════════════════

_app = None

def run_agent(query: str, session_id: str = "default", model_name: str = None, embedding_model: str = None, **kwargs):
    global _app
    if not _app:
        _app = create_workflow()
        
    initial_state = {
        "query": query,
        "messages": [{"role": "user", "content": query}],
        "next_agent": "orchestrator",
        "worker_model": model_name or "glm-4.7-flash", # 워커 모델 명시
        "orchestrator_model": "gpt-4o-mini", # 오케스트레이터 모델 명시
        "model_name": model_name # 하위 호환성 유지
    }
    

    # LangGraph 실행 (무한 루프 방지를 위해 recursion_limit 설정 - 복합 질문 처리를 위해 20으로 상향)
    result = _app.invoke(initial_state, config={"recursion_limit": 20})

    
    return {
        "answer": result.get("final_answer", "답변을 생성하지 못했습니다."),
        "wrapper": True # 호환성
    }


# ═══════════════════════════════════════════════════════════════════════════
# 외부 노출 도구 목록
# ═══════════════════════════════════════════════════════════════════════════

AGENT_TOOLS = [
    search_sop_tool,
    get_version_history_tool,
    compare_versions_tool,
    get_references_tool
]
