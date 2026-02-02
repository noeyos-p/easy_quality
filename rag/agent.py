"""
SOP 에이전트 모듈 v12.8 (Hybrid Deep Search)

🤖 하이브리드 ReAct 멀티 에이전트
- 하이브리드 검색: Vector(의미) + SQL(키워드) + Graph(참조) 결합
- 검색 누락 방지: 벡터 검색 임계값 최적화(0.20) 및 SQL 기반 전역 폴백 검색
- 전문 답변 보장: 내부 규정 기반 상세 검증 보고서 레이아웃 고정
"""

import os
from typing import List, Dict, Optional, Any, Annotated, TypedDict
from datetime import datetime
import operator
import re
import json

# ═══════════════════════════════════════════════════════════════════════════
# 임포트 및 설정
# ═══════════════════════════════════════════════════════════════════════════

if os.getenv("LANGCHAIN_API_KEY"):
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "gmp-sop-agent")
    # 🔥 LangSmith에서 최상위 프로젝트 및 런 네임 강제 지정
    from langchain_core.tracers.context import tracing_v2_enabled

try:
    from zai import ZaiClient
    ZAI_AVAILABLE = True
except ImportError:
    ZAI_AVAILABLE = False

try:
    from langchain_core.tools import tool
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

try:
    from langgraph.graph import StateGraph, START, END
    LANGGRAPH_AGENT_AVAILABLE = True
except ImportError:
    LANGGRAPH_AGENT_AVAILABLE = False

# 🔥 LangSmith 가시성 강화를 위한 추적기 임포트
try:
    from langsmith import traceable
except ImportError:
    import functools
    def traceable(name=None, run_type=None, **kwargs):
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs): return func(*args, **kwargs)
            return wrapper
        return decorator

# ═══════════════════════════════════════════════════════════════════════════
# 하이브리드 도구 정의
# ═══════════════════════════════════════════════════════════════════════════

_vector_store = None
_graph_store = None
_sql_store = None

def init_agent_tools(vector_store_module, graph_store_instance, sql_store_instance=None):
    global _vector_store, _graph_store, _sql_store
    _vector_store = vector_store_module
    _graph_store = graph_store_instance
    _sql_store = sql_store_instance

@tool
def hybrid_search_sop(query: str, keywords: List[str] = None, embedding_model: str = "intfloat/multilingual-e5-small") -> str:
    """SOP 문서 검색 (Vector + SQL 하이브리드)
    keywords: SQL 검색에 사용할 핵심 단어 목록 (LLM이 미리 정제한 값)
    """
    global _sql_store
    if not _vector_store: return "❌ 벡터 스토어 미설정"
    
    combined_results = []
    seen_ids = set()
    
    # 1. 지능형 키워드 및 숫자 파편 추출 (강화된 Regex)
    search_terms = keywords or []
    # 질문이나 키워드에서 모든 숫자 덩어리를 추출하여 잠재적 ID로 취급
    all_numbers = re.findall(r'\d+', f"{query} {' '.join(search_terms)}")
    search_terms.extend(all_numbers)
    
    # 중복 제거 및 정규화
    unique_terms = list(set([str(k).upper().strip() for k in search_terms if k]))
    is_summary_request = any(word in query for word in ["요약", "정리", "summary", "전체", "리뷰", "본문"])

    print(f"📡 [HybridSearch] SQL 저장소 상태: {'Connected' if _sql_store else 'Disconnected'}")
    print(f"📡 [HybridSearch] 추출된 지능형 키워드: {unique_terms}")

    # [우선순위 1] SQL 저장소 전수 조사 (전체 본문 및 고정 ID 매칭)
    if _sql_store:
        try:
            all_docs = _sql_store.list_documents()
            for doc in all_docs:
                sop_id = doc.get("sop_id", "")
                if not sop_id: continue
                
                match_found = False
                # 키워드 중 하나라도 ID의 일부이거나, ID가 키워드를 포함하는지 확인
                for kw in unique_terms:
                    if kw in sop_id.upper() or sop_id.upper() in kw:
                        match_found = True
                        break
                
                if match_found and sop_id not in seen_ids:
                    full_doc = _sql_store.get_document_by_id(sop_id)
                    if full_doc:
                        content = full_doc.get("markdown_content", "")
                        if content and len(content.strip()) > 50:
                            # 요약 시에는 대용량 컨텍스트 (최대 15000자) 제공
                            limit = 15000 if is_summary_request else 4000
                            combined_results.append(f"📄 [전역 검색/원본 본문] 출처: {sop_id}\n{content[:limit]}")
                            seen_ids.add(sop_id)
                            print(f"✅ [HybridSearch] SQL 매칭 성공: {sop_id}")
        except Exception as e:
            print(f"⚠️ [SQL Search Error] {e}")

    # [우선순위 2] 벡터 검색 (의미 중심 파편 찾기)
    # 요약 요청 시 검색된 본문이 있으면 파편 정보는 생략하여 토큰 절약
    if not is_summary_request or len(combined_results) == 0:
        try:
            results = _vector_store.search(
                query=query, 
                collection_name="documents", 
                n_results=15,
                model_name=embedding_model,
                similarity_threshold=0.12 # 벡터 검색 임계값 대폭 완화
            )
            for r in results:
                meta = r.get("metadata", {})
                text = r.get("text", "")
                doc_id = meta.get('sop_id', 'N/A')
                
                # 이미 전체 본문을 가져온 문서의 파편은 스킵
                if doc_id in seen_ids: continue
                
                source = f"[{doc_id}] > {meta.get('section_path', '')} (p.{meta.get('page', 'N/A')})"
                combined_results.append(f"📄 출처: {source}\n{text}")
        except Exception: pass

    # 만약 아무것도 못 찾았다면, 모든 문서 요약 시도 예외 처리
    if not combined_results and is_summary_request and _sql_store:
        try:
            # 질문에서 제목이나 ID를 유추하지 못했을 때 마지막 시도로 목록의 첫 번째 문서라도 가져옴
            all_docs = _sql_store.list_documents()
            if all_docs:
                 doc = all_docs[0] # 임시: 첫 번째 문서
                 full_doc = _sql_store.get_document_by_id(doc['sop_id'])
                 combined_results.append(f"📄 [전역 검색/폴백] 출처: {doc['sop_id']}\n{full_doc.get('markdown_content', '')[:10000]}")
        except: pass

    return "\n\n".join(combined_results)

@tool
def get_document_references(sop_id: str) -> str:
    """참조 문서 조회 (Graph)"""
    if not _graph_store: return ""
    try:
        refs = _graph_store.get_document_references(sop_id.upper())
        if not refs: return ""
        doc = refs.get("document", {})
        return f"📄 {doc.get('sop_id')} 참조문서: {', '.join(refs.get('references', []))}"
    except Exception: return ""

AGENT_TOOLS = [hybrid_search_sop, get_document_references]

# ═══════════════════════════════════════════════════════════════════════════
# 에이전트 시스템 프롬프트
# ═══════════════════════════════════════════════════════════════════════════

# 공통 역할 지정
BASE_SYSTEM_PROMPT = """당신은 회사 내부 GMP 규정(SOP) 전문가입니다.
당신의 모든 답변은 **오직 제공된 규정 데이터(Tool Observation)**에만 기반해야 합니다.

## 🚫 절대 금지 사항 (Hallucination Warning)
1. **외부 지식 사용 금지**: 당신이 이미 알고 있는 일반적인 GMP 지식(예: ALCOA+, 일반적인 로그북 정의 등)이라도 검색된 데이터에 명시되어 있지 않다면 절대로 답변에 포함하지 마세요.
2. **추측 금지**: 데이터에 없는 내용을 "일반적으로 ~하다"라고 추측하여 답변하지 마세요.
3. **증거 없는 답변 금지**: 검색된 텍스트 조항에서 직접적인 근거를 찾을 수 없는 내용은 누락시키세요.

## 📋 답변 원칙
1. **증거 기반 (Strictly Evidence-based)**: 답변의 모든 문장은 제공된 SOP 텍스트의 특정 조항에서 기인해야 합니다.
2. **정확한 인용**: 답변 시 반드시 `[SOP ID] > [섹션명] (p.페이지)` 형식으로 출처를 명시하세요.
3. **데이터 누락 시**: 관련 내용이 검색되지 않았다면 "제공된 규정에서 해당 내용을 찾을 수 없습니다"라고 정직하게 답변하세요.
"""

# 전문 진단용 추가 지침 (OOS, 일탈, 재시험 등 복잡한 상황)
VERIFICATION_INSTRUCTIONS = """
## 🎯 핵심 원칙: 능동적 추론 (Active Reasoning)
1. **명시적 규정 우선**: 문서에 직설적으로 "금지" 또는 "허용"이 명시된 경우 이를 최우선으로 합니다.
2. **논리적 추론 (Deduction)**: 구체적인 허용 여부가 없더라도, 상위 규정의 취지를 바탕으로 하위 상황을 논리적으로 추론하여 결론을 도출하세요. "규정의 취지상 ~해야 한다"는 방향을 제시하세요.

## 📋 답변 구조 (필수: 보고서 형식)
### **1. 검증 의견**
- [**핵심 결론**]: 결론을 한 문장으로 명확히 제시
- [**상세 분석**]: 규정의 취지와 사용자 상황을 대조하여 논리적으로 설명

### **2. 검증 근거 및 출처**
- 각 근거별 번호와 제목
- 정확한 출처 표기 필수: `**[출처]** [SOP ID] > [제목] > [상세 문구 인용] (p.페이지)`

### **3. 조치 권고 및 제언**
- 사용자가 즉시 취해야 할 행동 가이드 및 관련 하위 지침서 제안
"""

# 단순 정보 제공용 추가 지침 (요약, 설명 등)
INFO_INSTRUCTIONS = """
## 📋 답변 구조
- 사용자의 요청에 대해 **문서에 명시된 텍스트를 충실히 반영**하여 답변하세요.
- AI의 수려한 요약보다 **문서상의 정확한 정의와 요건**을 나열하는 것이 더 중요합니다.
- 답변의 각 주요 항목 끝에는 반드시 구체적인 근거 조항을 명시하세요.
- **다시 한 번 강조**: 검색 결과에 없는 개념(예: ALCOA+ 등)을 외부 지식으로 보충하지 마세요. 오직 "문서 안에서 찾은 결과"만 보여주세요.
"""

class AgentState(TypedDict):
    query: str
    model_name: str
    embedding_model: str
    search_results: List[Dict]
    answer: str
    reasoning: str
    queries: List[str]
    keywords: List[str]
    is_verification: bool # 추가: 검증 성격 질문 여부
    # 🔥 ReAct 루프를 위한 메시지 기록 추가
    messages: Annotated[List[Any], operator.add]

_agent = None

def create_agent(model_name: str = "glm-4.7-flash"):
    global _agent
    api_key = os.getenv("ZAI_API_KEY")
    if not api_key: raise ValueError("ZAI_API_KEY 설정 필요")
    _agent = {"model": model_name, "client": ZaiClient(api_key=api_key)}
    return _agent

@traceable(run_type="llm", name="Z.AI-LLM-Completion")
def _llm_chat_completion(messages: List[Dict], model: str, tools: Optional[List] = None, tool_choice: str = "auto"):
    """LangSmith에서 LLM 노드로 표시되도록 하는 추적용 래퍼"""
    return _agent["client"].chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,
        tool_choice=tool_choice,
        temperature=0.1
    )

def query_expansion_node(state: AgentState):
    """멀티 쿼리 및 지능형 키워드 확장 (Regex 방지)"""
    client = _agent["client"]
    print(f"🧠 [Agent] 검색 전략 수립 중 (Query: {state['query']})")
    
    prompt = f"""사용자 질문을 분석하여 규정 검색을 위한 전략을 수립하세요.
특히 "10번 문서"와 같이 숫자가 언급되면 이는 "EQ-SOP-00010"과 같은 관리 번호의 파편일 가능성이 매우 높으므로, 해당 숫자를 키워드에 포함하세요.

[사용자 질문]
{state['query']}

결과는 반드시 다음과 같은 JSON 형식으로만 답변하세요.
{{
  "expanded_queries": ["검색용으로 확장된 문장 2개"],
  "keywords": ["SQL 검색에 사용할 핵심 단어/숫자 5개 (ID 포함)"]
}}"""
    
    try:
        res = _llm_chat_completion(
            model=state["model_name"], 
            messages=[{"role": "user", "content": prompt}]
        )
        content = res.choices[0].message.content
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        import json
        if json_match:
            data = json.loads(json_match.group(0))
            return {
                "queries": [state["query"]] + data.get("expanded_queries", []),
                "keywords": data.get("keywords", [])
            }
    except Exception as e:
        print(f"⚠️ [Expansion] 실패: {e}")
    
    return {"queries": [state["query"]], "keywords": []}

def reasoner_node(state: AgentState):
    """사고(Reasoning) 및 행동(Acting) 결정 노드"""
    print(f"⚖️ [Reasoner] 사고 중... (Message Count: {len(state['messages'])})")
    
    # 질문 성격에 따른 시스템 프롬프트 선택
    is_verification = any(kw in state['query'] for kw in ["되나요", "가능한가요", "위반", "적절", "판단", "허용", "금지", "적합"])
    instructions = VERIFICATION_INSTRUCTIONS if is_verification else INFO_INSTRUCTIONS
    
    system_prompt = f"{BASE_SYSTEM_PROMPT}\n{instructions}"
    
    # 상태 업데이트용 사전 정보
    state_update = {"is_verification": is_verification}
    
    # 도구 정의 전달 (LLM이 도구 사용 여부 결정)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "hybrid_search_sop",
                "description": "SOP 문서 검색 (Vector + SQL 하이브리드). 요약이 필요하거나 특정 번호 조회가 필요할 때 유용합니다.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "검색어"},
                        "keywords": {"type": "array", "items": {"type": "string"}, "description": "SQL 매칭용 핵심 키워드 목록"}
                    },
                    "required": ["query"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_document_references",
                "description": "특정 SOP 문서가 참조하고 있는 다른 연관 문서 목록을 조회합니다 (Graph DB 기반).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "sop_id": {"type": "string", "description": "조회할 SOP ID (예: EQ-SOP-00010)"}
                    },
                    "required": ["sop_id"]
                }
            }
        }
    ]

    messages = [{"role": "system", "content": system_prompt}] + state["messages"]
    
    try:
        res = _llm_chat_completion(
            model=state["model_name"], 
            messages=messages, 
            tools=tools,
            tool_choice="auto"
        )
        msg = res.choices[0].message
        
        # 도구 호출이 있는 경우
        if msg.tool_calls:
            return {"messages": [msg]}
        
        # 최종 답변인 경우
        return {
            **state_update,
            "messages": [msg],
            "answer": msg.content or "",
            "reasoning": getattr(msg, 'reasoning_content', "")
        }
    except Exception as e:
        print(f"⚠️ [Reasoner Error] {e}")
        return {"messages": [{"role": "assistant", "content": f"❌ 오류 발생: {e}"}]}

def tool_node(state: AgentState):
    """도구 실행(Acting) 및 결과(Observation) 반환 노드"""
    last_msg = state["messages"][-1]
    new_messages = []
    
    if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
        for tc in last_msg.tool_calls:
            tool_name = tc.function.name
            args = json.loads(tc.function.arguments)
            
            print(f"🛠️ [Tool] {tool_name} 실행 중... ({args})")
            
            obs = ""
            if tool_name == "hybrid_search_sop":
                kw = args.get("keywords") or state.get("keywords", [])
                obs = hybrid_search_sop.invoke({
                    "query": args["query"], 
                    "keywords": kw,
                    "embedding_model": state.get("embedding_model")
                })
            elif tool_name == "get_document_references":
                obs = get_document_references.invoke({
                    "sop_id": args["sop_id"]
                })
            
            new_messages.append({
                "role": "tool", 
                "tool_call_id": tc.id, 
                "name": tool_name, 
                "content": obs or "❌ 검색 결과가 없습니다."
            })
    
    return {"messages": new_messages}

def verifier_node(state: AgentState):
    """최종 규정 검증 및 무결성 체크 노드"""
    print(f"⚖️ [Verifier] 최종 규정 적합성 판단 및 검증 중")
    
    # 시스템 프롬프트 개편: 정답 우선 원칙
    is_v = state.get("is_verification", False)
    
    if is_v:
        # 검증 모드: 심층 보고서 + 정답
        verification_prompt = f"""당신은 품질보증(QA) 부서의 최종 승인권자입니다. 
사용자의 질문에 대해 규정 근거를 바탕으로 **적합성 판정 및 최종 답변**을 작성하세요.

## 🎯 답변 작성 가이드 (검증 모드)
1. **결론 (Conclusion)**: 질문에 대한 적합성 여부(예: 허용/금지/위반 등)를 최상단에 명확히 기재하세요.
2. **상세 근거**: 검색된 SOP의 조항들을 인용하여 왜 그런 결론이 나왔는지 논리적으로 설명하세요.
3. **QA 검토 보고서**: 답변 하단에 [상충/누락/준수] 여부를 포함한 보고서 섹션을 구성하세요.

사용자 질문: {state.get('query')}"""
    else:
        # 정보 검색 모드: 정답 중심 + 최소한의 검증
        verification_prompt = f"""당신은 GMP 규정(SOP) 안내 전문가입니다. 
사용자가 찾는 정보를 검색된 데이터에서 추출하여 **친절하고 정확하게** 답변하세요.

## 🎯 답변 작성 가이드 (정보 검색 모드)
1. **직접적인 정답**: 사용자가 묻는 정보(예: 특정 조항의 내용, 문서 번호 등)를 가장 먼저, 명확하게 답변하세요. 
2. **불필요한 형식 지양**: "QA 검토 결과"와 같은 딱딱한 보고서 형식을 최상단에 두지 마세요. 
3. **근거 표기**: 답변 내용 끝에 출처(조항 번호)만 짧게 덧붙이세요.

사용자 질문: {state.get('query')}"""

    messages = [{"role": "system", "content": verification_prompt}] + state["messages"]
    
    try:
        # 검증 결과 생성 (LLM 호출)
        res = _llm_chat_completion(
            model=state["model_name"], 
            messages=messages,
            tool_choice="none" # 최종 답변 생성 시에는 도구 사용 안 함
        )
        msg = res.choices[0].message
        return {
            "answer": msg.content or "",
            "reasoning": getattr(msg, 'reasoning_content', ""),
            "messages": [msg]
        }
    except Exception as e:
        print(f"⚠️ [Verifier Error] {e}")
        return {}

def should_continue(state: AgentState):
    """루프 종료 여부 결정"""
    last_msg = state["messages"][-1]
    if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
        return "tools"
    return END

# ═══════════════════════════════════════════════════════════════════════════
# 🤖 LangGraph 워크플로우 정의
# ═══════════════════════════════════════════════════════════════════════════

def create_workflow():
    if not LANGGRAPH_AGENT_AVAILABLE:
        return None
    
    workflow = StateGraph(AgentState)
    
    # 노드 추가
    workflow.add_node("expansion", query_expansion_node)
    workflow.add_node("reasoner", reasoner_node)
    workflow.add_node("tools", tool_node)
    workflow.add_node("verifier", verifier_node) # 검증 노드 추가
    
    # 엣지 연결
    workflow.add_edge(START, "expansion")
    workflow.add_edge("expansion", "reasoner")
    
    # 🔥 ReAct 루프: Reasoner -> (Tools -> Reasoner) -> Verifier -> End
    workflow.add_conditional_edges(
        "reasoner",
        should_continue,
        {
            "tools": "tools",
            # 단순 정보 검색인 경우 검증 노드를 건너뛰거나 바로 엔드로 갈 수도 있지만,
            # 현재는 모든 최종 답변의 품질을 위해 verifier를 거치되 프롬프트로 제어함
            END: "verifier" 
        }
    )
    workflow.add_edge("tools", "reasoner")
    workflow.add_edge("verifier", END)
    
    return workflow.compile()

# 전역 그래프 인스턴스
_workflow_app = None

def run_agent(query: str, session_id: str = "default", model_name: str = "glm-4.7-flash", embedding_model: str = "intfloat/multilingual-e5-small"):
    global _workflow_app
    if not _agent: create_agent(model_name)
    
    # 🔥 LangSmith에서 "Input" 컬럼에 사용자의 질문이 명확하게 나오도록 래핑 함수 정의
    @traceable(name="GMP-SOP-Orchestrator", run_type="chain")
    def _orchestrated_run(user_input: str, state: dict, runner_config: dict):
        return _workflow_app.invoke(state, config=runner_config)

    initial_state = {
        "query": query, 
        "model_name": model_name, 
        "embedding_model": embedding_model,
        "queries": [],
        "keywords": [],
        "search_results": [],
        "messages": [{"role": "user", "content": query}] # 메시지 초기화
    }
    
    # LangGraph를 통한 실행
    if LANGGRAPH_AGENT_AVAILABLE:
        if not _workflow_app:
            _workflow_app = create_workflow()
        
        # 명시적인 run_name 부여로 LangSmith 가시성 확보
        config = {
            "configurable": {"thread_id": session_id},
            "run_name": "GMP-SOP-Orchestrator",
            "metadata": {"session_id": session_id, "model": model_name}
        }
        
        # user_input을 명시적으로 전달하여 LangSmith 가시성 고정
        final_state = _orchestrated_run(user_input=query, state=initial_state, runner_config=config)
    else:
        # 폴백: 수동 노드 호출
        expanded = query_expansion_node(initial_state)
        initial_state.update(expanded)
        final_state = verifier_agent_node(initial_state)
    
    return {
        "answer": final_state.get("answer", "❌ 답변을 생성하지 못했습니다."), 
        "reasoning": final_state.get("reasoning", ""),
        "success": True,
        "tool_calls": [{"tool": "hybrid_search", "queries": final_state.get("queries")}]
    }
