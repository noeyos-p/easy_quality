"""
SOP 멀티 에이전트 시스템 v13.0
- Orchestrator (Main): OpenAI (GPT-4o) - 질문 분석 및 라우팅, 최종 답변
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
    extract_english: True면 영문 내용 위주로 추출
    """
    global _sql_store, _vector_store
    
    results = []
    seen_ids = set()
    
    # 1. SQL 키워드 검색
    if _sql_store and keywords:
        all_docs = _sql_store.list_documents()
        for doc in all_docs:
            sop_doc = _sql_store.get_document_by_id(doc['sop_id'])
            if not sop_doc: continue
            
            content = sop_doc.get("markdown_content", "")
            # 키워드 매칭 (단순화)
            if any(k.upper() in doc['sop_id'].upper() for k in keywords):
                if extract_english:
                    # 영문 추출 로직 (알파벳 비율이 높은 문단만 필터링)
                    eng_paras = [p for p in content.split('\n\n') if len(re.findall(r'[a-zA-Z]', p)) > len(re.findall(r'[가-힣]', p))]
                    results.append(f"📄 [영문 발췌] {doc['sop_id']}:\n" + "\n".join(eng_paras[:5]))
                else:
                    results.append(f"📄 [전체 본문] {doc['sop_id']}:\n{content[:3000]}...")
                seen_ids.add(doc['sop_id'])
    
    # 2. 벡터 검색
    if _vector_store:
        vec_res = _vector_store.search(query, n_results=5)
        for r in vec_res:
            meta = r.get('metadata', {})
            sop_id = meta.get('sop_id')
            if sop_id not in seen_ids:
                results.append(f"📄 [검색] {sop_id} > {meta.get('section', '')}:\n{r.get('text', '')}")
                
    return "\n\n".join(results) if results else "검색 결과 없음"

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
    model_name: Optional[str] # 동적 모델 선택을 위한 필드 추가

# ═══════════════════════════════════════════════════════════════════════════
# 노드 정의 (Nodes)
# ═══════════════════════════════════════════════════════════════════════════

def orchestrator_node(state: AgentState):
    """
    메인 에이전트 (OpenAI GPT-4o)
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
    
    [에이전트 목록]
    1. retrieval: 규정 검색, 정보 조회 (SQL:키워드/ID우선 -> Vector:의미검색)
    2. summary: 문서나 조항의 요약 (SQL 원문 조회 후 요약)
    3. comparison: 구버전/신버전 비교 (SQL 버전 히스토리)
    4. graph: 참조/인용 관계 확인 (Graph DB)
    
    [출력 형식]
    JSON 형식으로 'next_action' (agent 이름 또는 'finish')과 'reason'을 반환하세요.
    예: {"next_action": "retrieval", "reason": "규정 검색 결과가 부족하여 재검색 필요"}
    """
    
    # 마지막 메시지가 도구 결과(Context)라면 답변 생성 모드로 진입 확률 높음
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
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
                model="gpt-4o",
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
        model=state.get("model_name") or "glm-4.7-flash", # 동적 모델 적용
        messages=[{"role": "user", "content": prompt}]
    )
    
    return {"messages": [{"role": "assistant", "content": f"[검색 에이전트 보고]\n{res.choices[0].message.content}"}]}

def summary_agent_node(state: AgentState):
    """[서브] 요약 에이전트 (Z.AI)"""
    client = get_zai_client()
    query = state["query"]
    
    search_res = search_sop_tool.invoke({"query": query})
    
    res = client.chat.completions.create(
        model=state.get("model_name") or "glm-4.7-flash", # 동적 모델 적용
        messages=[
            {"role": "system", "content": "문서를 요약하여 핵심 내용을 글머리 기호로 정리하세요."},
            {"role": "user", "content": f"질문: {query}\n\n문서 내용:\n{search_res}"}
        ]
    )
    return {"messages": [{"role": "assistant", "content": f"[요약 에이전트 보고]\n{res.choices[0].message.content}"}]}

def comparison_agent_node(state: AgentState):
    """[서브] 비교 에이전트 (Z.AI)"""
    client = get_zai_client()
    query = state["query"]
    
    # ID 및 버전 추출 로직은 복잡하므로 간단히 가정
    # 예: "SOP-001 버전 1.0과 2.0 비교해줘" -> 정규식으로 추출 필요
    # 여기서는 데모용으로 하드코딩된 로직 대신 LLM에게 추출 유도 가능
    
    res = client.chat.completions.create(
        model=state.get("model_name") or "glm-4.7-flash", # 동적 모델 적용
        messages=[{"role": "user", "content": f"사용자 질문 '{query}'에서 문서 ID와 버전 두 개를 추출해서 JSON으로 줘. 형식: {{'id': '...', 'v1': '...', 'v2': '...'}} "}]
    )
    try:
        info = json.loads(res.choices[0].message.content)
        comp_res = compare_versions_tool.invoke({"sop_id": info['id'], "v1": info['v1'], "v2": info['v2']})
        
        final_res = client.chat.completions.create(
            model=state.get("model_name") or "glm-4.7-flash", # 동적 모델 적용
            messages=[{"role": "user", "content": f"두 버전의 차이점을 분석해줘:\n{comp_res}"}]
        )
        content = final_res.choices[0].message.content
    except:
        content = "버전 정보를 정확히 추출하지 못했습니다. (예: SOP-001 1.0과 2.0 비교해줘)"

    return {"messages": [{"role": "assistant", "content": f"[비교 에이전트 보고]\n{content}"}]}

def graph_agent_node(state: AgentState):
    """[서브] 그래프 에이전트 (Z.AI)"""
    client = get_zai_client()
    query = state["query"]
    
    # SOP ID 추출
    match = re.search(r'([A-Z]{2}-SOP-\d+)', query.upper())
    if match:
        sop_id = match.group(1)
        refs = get_references_tool.invoke({"sop_id": sop_id})
        content = f"문서 {sop_id}의 참조 관계입니다:\n{refs}"
    else:
        content = "문서 ID를 찾을 수 없습니다."
        
    return {"messages": [{"role": "assistant", "content": f"[그래프 에이전트 보고]\n{content}"}]}

# ═══════════════════════════════════════════════════════════════════════════
# 워크플로우 구성
# ═══════════════════════════════════════════════════════════════════════════

def create_workflow():
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
        "model_name": model_name # 모델 정보 주입
    }
    
    # LangGraph 실행
    result = _app.invoke(initial_state)
    
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
