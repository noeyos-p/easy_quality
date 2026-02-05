"""
문서 검색 서브에이전트 모듈 (Deep Agent 스타일)
- 질문 분석 후 다단계 정밀 검색을 수행하는 그래프 구조의 에이전트
- 벡터 검색 (Weaviate), SQL 검색 (PostgreSQL) 통합
"""

import os
import re
import json
import hashlib
import operator
from typing import Any, Dict, List, Optional, Literal, Annotated, TypedDict

from langsmith import traceable
from langgraph.graph import StateGraph, START, END

# ═══════════════════════════════════════════════════════════════════════════
# 전역 스토어 및 클라이언트 관리
# ═══════════════════════════════════════════════════════════════════════════

_vector_store = None
_sql_store = None
_graph_store = None
_zai_client = None

def init_search_stores(vector_store_module=None, sql_store_instance=None, graph_store_instance=None):
    """검색 에이전트용 스토어 초기화"""
    global _vector_store, _sql_store, _graph_store
    _vector_store = vector_store_module
    _sql_store = sql_store_instance
    _graph_store = graph_store_instance

def get_zai_client():
    """Z.AI 클라이언트 반환"""
    global _zai_client
    if not _zai_client:
        from backend.agent import get_zai_client as get_main_zai
        _zai_client = get_main_zai()
    return _zai_client

# ═══════════════════════════════════════════════════════════════════════════
# 핵심 검색 로직
# ═══════════════════════════════════════════════════════════════════════════

def _get_clause_and_doc_from_db(content: str, metadata: dict) -> tuple:
    """
    벡터 DB metadata 또는 SQL DB에서 문서명과 조항 정보를 가져옵니다.

    Returns:
        (doc_name, clause): 문서명과 조항 정보 튜플
    """
    global _sql_store

    # 1. metadata에서 먼저 확인 (우선순위: doc_id > doc_id > doc_name)
    doc_name = metadata.get('doc_id') or metadata.get('doc_id') or metadata.get('doc_name')

    # 조항 정보 확인 (우선순위: clause_id > clause > section > ...)
    # "본문", "전체" 같은 의미 없는 값은 무시
    invalid_clause_values = ["본문", "전체", "전문", "None", ""]
    clause_keys = ['clause_id', 'clause', 'section', 'article', 'article_num', 'section_id', '조항']
    
    for key in clause_keys:
        value = metadata.get(key)
        if value and str(value).strip() not in invalid_clause_values:
            clause = str(value)
            
            # [보정] CH형태의 기술적 ID나 너무 긴 ID는 제거 또는 정제
            if re.search(r'CH\d+', clause) or len(clause) > 20:
                # 제목(title)이 있다면 제목으로 대체 시도
                title = metadata.get('title') or metadata.get('current_title')
                if title:
                    return (doc_name or "Unknown", title)
                return (doc_name or "Unknown", "상세")
                
            return (doc_name or "Unknown", clause)

    # 2. SQL DB에서 content 기반으로 역으로 찾기
    if _sql_store:
        try:
            # content의 고유한 부분 추출 (앞 100자)
            content_sample = content[:100].strip()

            # 모든 문서 조회
            all_docs = _sql_store.list_documents()

            for doc in all_docs:
                doc_id = doc.get('id')
                chunks = _sql_store.get_chunks_by_document(doc_id)

                for chunk in chunks:
                    chunk_content = chunk.get('content', '').strip()
                    # content 매칭 (포함 관계 확인)
                    if content_sample in chunk_content or chunk_content[:100] in content:
                        found_doc_name = doc.get('doc_name', 'Unknown')
                        found_clause = chunk.get('clause') or '본문'
                        print(f"    [SQL 역조회] 발견: {found_doc_name} - {found_clause}")
                        return (found_doc_name, found_clause)
        except Exception as e:
            print(f" SQL 역조회 실패: {e}")

    return (doc_name or "Unknown", "본문")

def search_documents_internal(
    query: str,
    max_results: int = 10,  # 검색 수량 확대 (기존 5 -> 10)
    search_type: Literal["hybrid", "vector", "keyword"] = "hybrid",
    keywords: List[str] = None,
    target_clause: str = None, # 조항 번호 직접 조회 (Point Lookup)
) -> List[Dict[str, Any]]:
    """내부용 검색 실행 함수"""
    global _vector_store, _sql_store
    results = []
    seen_content = set()

    # 0. 조항 번호 직접 및 하위 조회 (SQL Point & Prefix Match)
    if target_clause and _sql_store:
        try:
            print(f"    [Point/Prefix Lookup] 조항 및 하위 조항 조회 시도: {target_clause}")
            all_docs = _sql_store.list_documents()
            for doc in all_docs:
                doc_id = doc.get('id')
                chunks = _sql_store.get_chunks_by_document(doc_id)
                
                # 조항 번호가 정확히 일치하거나 해당 조항의 하위(예: 5.4.2 -> 5.4.2.1)인 경우 모두 포함
                sub_chunks = []
                for chunk in chunks:
                    clause_val = str(chunk.get('clause'))
                    if clause_val == target_clause or clause_val.startswith(f"{target_clause}."):
                        content = chunk.get('content', '')
                        content_hash = hashlib.md5(content.encode()).hexdigest()
                        if content_hash not in seen_content:
                            sub_chunks.append({
                                "doc_name": doc.get('doc_name', 'Unknown'),
                                "section": clause_val,
                                "content": content,
                                "source": "sql-hierarchical-lookup",
                                "score": 2.0, # 직접/하위 매칭은 최고 점수
                                "hash": content_hash
                            })
                            seen_content.add(content_hash)
                
                # 조항이 발견되었을 경우, 검색 결과에 추가
                results.extend(sub_chunks)
        except Exception as e:
            print(f" Hierarchical lookup failed: {e}")

    # 1. 벡터/하이브리드 검색 및 컨텍스트 확장
    if _vector_store:
        try:
            enhanced_query = query
            if keywords:
                enhanced_query = f"{query} {' '.join(keywords)}"

            if search_type == "hybrid":
                current_alpha = 0.25 if keywords else 0.4
                vec_res = _vector_store.search_hybrid(enhanced_query, n_results=max_results * 2, alpha=current_alpha)
            else:
                vec_res = _vector_store.search(enhanced_query, n_results=max_results * 2)

            scored_results = []
            for r in vec_res:
                meta = r.get('metadata', {})
                content = r.get('text', '')
                if not content: continue

                content_hash = hashlib.md5(content.encode()).hexdigest()
                if content_hash in seen_content: continue
                
                doc_name, clause_info = _get_clause_and_doc_from_db(content, meta)
                
                # [부스팅] 조항 번호 매칭 가중치
                boost_score = r.get('similarity', 0)
                if keywords:
                    for kw in keywords:
                        if kw in clause_info or (meta.get('title') and kw in meta.get('title')):
                            boost_score += 0.5 
                
                if target_clause and (target_clause == clause_info or clause_info.startswith(f"{target_clause}.")):
                    boost_score += 1.0

                scored_results.append({
                    "doc_name": doc_name,
                    "section": clause_info,
                    "content": content,
                    "source": r.get('source', 'vector-hybrid'),
                    "score": boost_score,
                    "hash": content_hash,
                    "meta": meta # 확장 조회를 위해 메타 보관
                })

            scored_results.sort(key=lambda x: x["score"], reverse=True)
            
            # [지능형 확장] 상위 결과 중 내용이 제목뿐이거나 중요한 경우 다음 데이터 추가 로드
            for r in scored_results[:max_results]:
                if r["hash"] not in seen_content:
                    seen_content.add(r["hash"])
                    
                    # 제목성 청크(내용이 너무 짧음)인 경우 또는 점수가 매우 높은 경우 하위 내용 확장
                    if _sql_store and (len(r["content"]) < 100 or r["score"] > 0.8):
                        try:
                            doc_id_val = r["meta"].get("doc_id") or r["meta"].get("id")
                            if doc_id_val:
                                all_chunks = _sql_store.get_chunks_by_document(doc_id_val)
                                current_idx = -1
                                # 현재 청크의 위치 찾기
                                for idx, c in enumerate(all_chunks):
                                    if c.get("content") == r["content"]:
                                        current_idx = idx
                                        break
                                
                                # 다음 3개 청크를 '상세 내용'으로 병합
                                if current_idx != -1:
                                    extra_content = ""
                                    for i in range(1, 4):
                                        if current_idx + i < len(all_chunks):
                                            next_c = all_chunks[current_idx + i]
                                            extra_content += f"\n[상세] {next_c.get('content')}"
                                    
                                    if extra_content:
                                        r["content"] += extra_content
                                        print(f"    [Hierarchical Expansion] {r['section']} 하위 내용 확장 완료")
                        except Exception as ex:
                            print(f" Expansion error: {ex}")

                    results.append({
                        "doc_name": r["doc_name"],
                        "section": r["section"],
                        "content": r["content"][:4000],
                        "source": r["source"]
                    })
        except Exception as e:
            print(f" Vector search error: {e}")

    # 2. 관련 문서/조항으로 탐색 확장 (Graph DB 활용)
    # ... (생략 - 기존 로직 유지하되 results 필터링 반영)
    if _graph_store and results:
        try:
            extended_results = []
            # 상위 결과들에 대해 그래프 확장
            for r in results[:3]: 
                doc_name = r["doc_name"]
                refs = _graph_store.get_document_references(doc_name)
                if refs and refs.get("references"):
                    ref_list = refs["references"]
                    for ref_id in ref_list[:2]:
                        if _sql_store:
                            ref_doc = _sql_store.get_document_by_name(ref_id)
                            if ref_doc:
                                ref_content = ref_doc.get("content", "")
                                if ref_content and hashlib.md5(ref_content[:200].encode()).hexdigest() not in seen_content:
                                    extended_results.append({
                                        "doc_name": ref_id,
                                        "section": "참조 문서",
                                        "content": f"[참조 내용 명시] {ref_content[:1500]}...",
                                        "source": "graph-reference",
                                        "score": 0.5
                                    })
                                    seen_content.add(hashlib.md5(ref_content[:200].encode()).hexdigest())
            results.extend(extended_results)
        except Exception as e:
            print(f" Graph expansion error: {e}")

    return results

# ═══════════════════════════════════════════════════════════════════════════
# 딥 검색 에이전트 상태 정의 (CompiledSubAgent 호환)
# ═══════════════════════════════════════════════════════════════════════════

class SearchState(TypedDict):
    """
    가이드에 따른 에이전트 상태.
    'messages' 키를 포함하여 툴 호출과 응답 이력을 관리합니다.
    """
    messages: Annotated[List[Any], operator.add]
    query: str
    model: str
    final_answer: str

# ═══════════════════════════════════════════════════════════════════════════
# 노드 및 도구 설정
# ═══════════════════════════════════════════════════════════════════════════

def call_model_node(state: SearchState):
    """LLM이 질문을 분석하고 도구 호출 여부를 결정함 (자율 계획)"""
    client = get_zai_client()
    messages = state["messages"]
    
    # 딥 검색을 유도하는 시스템 프롬프트 (Best practices 반영)
    system_prompt = f"""당신은 GMP/SOP 전문 **Deep Search 에이전트**입니다.

    [역할]
    사용자의 질문에 답하기 위해 필요한 정보를 문서 저장소에서 찾아 정밀 분석합니다.

    [작업 가이드]
    1. **계획(Planning)**: 복합적인 질문인 경우, 질문을 여러 핵심 키워드나 특정 조항 번호로 분해하세요.
    2. **검색(Search)**: `search_documents_tool`을 사용하여 정보를 찾으세요.
       - **정밀 타겟팅**: 질문에 특정 조항 번호(예: 5.4.2)가 언급되었다면 `target_clause` 인자에 명시하세요.
       - **키워드 발췌**: 질문에서 '정의', '목적', '절차' 등 핵심 의도 단어를 발췌하여 `keywords`로 전달하세요.
    3. **검증 및 필터링 (중요)**: 
       - 검색된 결과 중 본문 없이 제목만 있는 조항(예: '3. 정의')이나 질문과 연관성이 낮은 데이터는 **철저히 무시**하세요.
       - 답변에 **실제로 사용한** 데이터 소스에 대해서만 답변 내용 중에 `[USE: 문서명 | 조항]` 형식의 태그를 남기세요. 
       - 이 태그가 없는 소스는 참고문헌에서 자동으로 제외됩니다.
    4. **답변 작성**: 검증된 정보를 바탕으로 자연스러운 **평문(Plain Text)** 답변을 작성하세요.

    [답변 작성 규칙]
    - 본문에서는 자연스럽게 내용만 설명하세요. 출처 표기는 하지 마세요.
    - 검색된 정보만을 사용하여 답변하세요.
    - 답변 끝에는 자동으로 [참고 문서] 섹션이 추가되므로, 직접 작성하지 마세요.
    - 답변 완료 시 반드시 [DONE] 태그를 붙여 작업 완료를 알리세요.

    [답변 형식 예시]
    작업지침서는 현장에서 수행되는 업무를 일관되게 운영하기 위한 지침 문서입니다.

    주요 특징은 다음과 같습니다.
    1. 부서 또는 공정 단위의 운영 흐름과 관리 방법을 규정합니다
    2. 세부적인 작업 방법보다는 기본적인 지침과 관리 기준을 제시합니다

    [DONE]

    [절대 금지 사항]
    - 본문 중간이나 끝에 [참고 문서] 섹션을 직접 작성하지 마세요 (자동 추가됨)
    - 본문에서 "[문서명 조항]" 형식의 출처를 표기하지 마세요
    - 볼드(**), 헤더(#), 리스트 기호(-, *), 이탤릭(_) 등 마크다운 형식 사용 금지
    - 강조가 필요한 경우 대괄호 [ ] 또는 줄바꿈을 활용하세요
    - 모든 텍스트는 일반 평문(Plain Text)으로만 작성해야 합니다
    """
    
    # 시스템 프롬프트를 메시지 맨 앞에 삽입
    full_messages = [{"role": "system", "content": system_prompt}] + messages
    
    # 도구 정의
    tools = [
        {
            "type": "function",
            "function": {
                "name": "search_documents_tool",
                "description": "GMP/SOP 문서를 검색합니다. 에이전트가 직접 검색 조건을 설계할 수 있습니다.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "검색 쿼리 (전체 문장 또는 핵심 문구)"},
                        "keywords": {
                            "type": "array", 
                            "items": {"type": "string"},
                            "description": "질문에서 발췌한 핵심 단어 리스트 (예: ['작업지침서', '절차'])"
                        },
                        "target_clause": {
                            "type": "string", 
                            "description": "콕 집어서 찾을 조항 번호 (예: '5.1.3', '5.4.2'). 명확할 때만 사용하세요."
                        }
                    },
                    "required": ["query"]
                }
            }
        }
    ]
    
    res = client.chat.completions.create(
        model=state["model"],
        messages=full_messages,
        tools=tools,
        tool_choice="auto"
    )
    
    return {"messages": [res.choices[0].message]}

def tool_executor_node(state: SearchState):
    """LLM이 요청한 도구를 실행하고 결과를 메시지에 추가함"""
    last_msg = state["messages"][-1]
    tool_calls = last_msg.tool_calls
    
    tool_outputs = []
    for tc in tool_calls:
        if tc.function.name == "search_documents_tool":
            args = json.loads(tc.function.arguments)
            query = args.get("query")
            keywords = args.get("keywords", [])
            target_clause = args.get("target_clause")
            print(f"    [Deep Search] 도구 호출: '{query}' (키워드: {keywords}, 타겟조항: {target_clause})")
            
            results = search_documents_internal(query=query, keywords=keywords, target_clause=target_clause)
            print(f"    [Deep Search] 검색 결과 {len(results)}건 발견")
            
            formatted_results = []
            for r in results:
                doc_name = r.get('doc_name', '알 수 없는 문서')
                section = r.get('section', '조항 미상')
                content = r.get('content', '')
                formatted_results.append(
                    f"[DATA_SOURCE]\n"
                    f"문서 정보: {doc_name}\n"
                    f"해당 조항: {section}\n"
                    f"본문 내용: {content}\n"
                    f"[END_SOURCE]"
                )
            
            content = "\n\n".join(formatted_results)
            if not content: content = "검색 결과가 없습니다."
            
            tool_outputs.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": content
            })
            
    return {"messages": tool_outputs}

# ═══════════════════════════════════════════════════════════════════════════
# 그래프 구성
# ═══════════════════════════════════════════════════════════════════════════

def create_deep_search_graph():
    """자율적 도구 호출을 수행하는CompiledSubAgent 스타일 그래프"""
    workflow = StateGraph(SearchState)
    
    workflow.add_node("agent", call_model_node)
    workflow.add_node("action", tool_executor_node)
    
    workflow.add_edge(START, "agent")
    
    def router(state: SearchState):
        last_msg = state["messages"][-1]
        if last_msg.tool_calls:
            return "action"
        return END
    
    workflow.add_conditional_edges("agent", router, {"action": "action", END: END})
    workflow.add_edge("action", "agent")
    
    return workflow.compile(name="Deep Search Agent Flow")

# ═══════════════════════════════════════════════════════════════════════════
# 참고문서 섹션 자동 생성
# ═══════════════════════════════════════════════════════════════════════════

def _ensure_reference_section(messages: List[Any], final_answer: str) -> str:
    """
    검색된 모든 문서의 정보를 추출하여 [참고 문서] 섹션을 무조건 추가합니다.

    Args:
        messages: 대화 메시지 이력 (tool 호출 결과 포함)
        final_answer: LLM이 생성한 최종 답변

    Returns:
        참고문서 섹션이 포함된 최종 답변
    """
    # 1. tool 메시지에서 문서 정보 추출
    referenced_docs = []
    seen = set()

    for msg in messages:
        # tool 역할의 메시지만 확인
        if isinstance(msg, dict) and msg.get("role") == "tool":
            content = msg.get("content", "")
        elif hasattr(msg, "role") and msg.role == "tool":
            content = msg.content
        else:
            continue

        # [DATA_SOURCE] 섹션 파싱
        sources = re.findall(
            r'\[DATA_SOURCE\]\s*문서 정보:\s*([^\n]+)\s*해당 조항:\s*([^\n]+)',
            content,
            re.MULTILINE
        )

        for doc_name, section in sources:
            doc_name = doc_name.strip()
            section = section.strip()

            # 중복 제거
            key = f"{doc_name}|{section}"
            if key not in seen:
                seen.add(key)
                referenced_docs.append((doc_name, section))

    # 2. 참고문헌 섹션 생성 (순수 동적 필터링)
    if referenced_docs:
        ref_section = "\n\n[참고 문서]\n"
        
        # 에이전트가 답변 본문에서 명시적으로 [USE: ...] 태그를 사용한 소스만 정밀 추출
        used_sources = re.findall(r'\[USE:\s*([^\|\]]+)\s*\|\s*([^\]]+)\]', final_answer)
        
        # [데이터 품질 검증] 알맹이 없는 '정의' 헤더 등을 필터링하는 로직 (하드코딩 제거)
        def is_useful_content(doc, sec):
            # 실질적으로 정보를 담고 있지 않은 제목성 청크는 제외
            # (여기서는 referenced_docs의 데이터를 다시 검증하거나, 
            # 에이전트가 명시한 것만 믿는 방식으로 단순화)
            return True

        used_map = {}
        for doc, sec in used_sources:
            d_key = doc.strip()
            s_key = sec.strip()
            if d_key not in used_map:
                used_map[d_key] = set()
            used_map[d_key].add(s_key)

        doc_map = {}
        for doc_name, section in referenced_docs:
            # 에이전트가 명시적으로 사용했다고 태그를 단 소스만 포함
            if doc_name in used_map and section in used_map[doc_name]:
                if doc_name not in doc_map:
                    doc_map[doc_name] = []
                # 중복 조항 방지
                if section not in doc_map[doc_name]:
                    doc_map[doc_name].append(section)
        
        # [최종 출력]
        if doc_map:
            for doc_name, sections in doc_map.items():
                unique_sections = sorted(list(sections))
                ref_section += f"- {doc_name} ({', '.join(unique_sections)})\n"
        else:
            # 에이전트가 태그를 하나도 사용하지 않은 경우
            ref_section = "" # 불필요한 참고문헌 섹션 노출 방지
    else:
        ref_section = ""

    # 3. 답변 본문에서 [USE: ...] 태그 제거 및 참고문서 섹션 추가
    final_answer_cleaned = re.sub(r'\[USE:\s*[^\]]+\]', '', final_answer).strip()
    
    # 기존 [참고 문서] 섹션 제거 로직 유지
    final_answer_cleaned = re.sub(
        r'\n*\[참고 문서\].*$',
        '',
        final_answer_cleaned,
        flags=re.DOTALL
    ).strip()

    return final_answer_cleaned + ref_section

# ═══════════════════════════════════════════════════════════════════════════
# 메인 엔트리 포인트
# ═══════════════════════════════════════════════════════════════════════════

_deep_search_app = None

@traceable(name="sub_agent:search")
def retrieval_agent_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """[서브] 검색 에이전트 (CompiledSubAgent 스타일)"""
    global _deep_search_app
    if not _deep_search_app:
        _deep_search_app = create_deep_search_graph()

    print(f" [Deep Search] 정밀 검색 가동: {state['query']}")

    initial_state = {
        "messages": [{"role": "user", "content": state["query"]}],
        "query": state["query"],
        "model": state.get("worker_model") or state.get("model_name") or "glm-4.7-flash",
        "final_answer": ""
    }

    # 내부 도구 호출 루프 실행 (재귀 한도 내에서 자율 검색)
    result = _deep_search_app.invoke(initial_state, config={"recursion_limit": 15})

    # 마지막 메시지가 LLM의 최종 답변
    final_msg = result["messages"][-1].content

    # 참고문서 섹션 자동 추가 (무조건 표시)
    final_msg_with_refs = _ensure_reference_section(result["messages"], final_msg)

    return {"messages": [{"role": "assistant", "content": final_msg_with_refs}]}

# ═══════════════════════════════════════════════════════════════════════════
# 레거시 도구 호환용 (필요 시)
# ═══════════════════════════════════════════════════════════════════════════

try:
    from langchain_core.tools import tool
except ImportError:
    def tool(func): return func

@tool
def search_sop_tool(query: str, extract_english: bool = False, keywords: List[str] = None) -> str:
    """SOP 문서 검색 도구 (레거시/내부용)"""
    search_query = query if not keywords else f"{query} {' '.join(keywords)}"
    results = search_documents_internal(query=search_query)
    
    if not results:
        return "검색 결과 없음."

    output = []
    for r in results:
        output.append(f"[검색] {r['doc_name']} > {r['section']}:\n{r['content']}")

    return "\n\n".join(output)
