"""
비교 에이전트 (Comparison Agent) - Refactored Version
- 두 문서 또는 같은 문서의 서로 다른 버전 간 차이점을 분석하거나 버전 목록을 조회합니다.
- 결과를 context에 보고서 형태로 저장합니다.
"""


import json
from typing import Optional
from backend.agent import get_zai_client, AgentState, safe_json_loads
# 순환 참조 방지를 위해 직접 Store 사용
from backend.sql_store import SQLStore
from backend.graph_store import Neo4jGraphStore

def normalize_version(v: Optional[str]) -> Optional[str]:
    """버전 번호 정량화: '1' -> '1.0', 'v1.1' -> '1.1'"""
    if not v: return v
    v_str = str(v).strip().lower()
    if v_str.startswith('v'): v_str = v_str[1:]
    # 도트가 없고 숫자만 있는 경우 .0 부착
    if '.' not in v_str and v_str.replace('.', '').isdigit():
        return f"{v_str}.0"
    return v_str

def comparison_agent_node(state: AgentState):
    """[서브] 비교 에이전트 - 버전 목록 조회 또는 내용 비교 분석"""
    print(f"🔵 [COMPARISON AGENT] 진입! query={state.get('query')}")
    client = get_zai_client()
    query = state["query"]
    model = state.get("worker_model") or state.get("model_name") or "glm-4.7-flash"
    
    # 1. 의도 분석 (Z.AI 활용)
    # 사용자가 버전 목록을 보고 싶어하는지, 아니면 실제 내용 비교를 원하는지 구분
    intent_prompt = f"""사용자의 질문을 분석하여 의도(Intent)와 필요한 정보를 추출하세요.
    - 질문: {query}
    
    [의도 분류]
    - list_history: 특정 문서의 버전 목록(히스토리)이 보고 싶을 때 (예: 버전 종류, 히스토리, 이력 등)
    - compare_versions: 두 버전 간의 내용을 구체적으로 비교하고 싶을 때. "최신 버전과 변경 내용", "차이점" 등을 물어보면 이에 해당합니다.
    
    반드시 JSON 형식으로만 답변하세요:
    {{"intent": "list_history" 또는 "compare_versions", "doc_id": "문서ID", "v1": "버전1(없으면 null)", "v2": "버전2(없으면 null)"}}
    """
    
    try:
        res = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": intent_prompt}],
            response_format={"type": "json_object"}
        )
        info = safe_json_loads(res.choices[0].message.content)
        intent = info.get("intent")
        doc_id = info.get("doc_id")
        v1 = normalize_version(info.get("v1"))
        v2 = normalize_version(info.get("v2"))
        
        print(f"👉 비교 의도: {intent}, 문서: {doc_id}, 버전: {v1} vs {v2}")

        # [CASE 1] 버전 목록 조회
        if intent == "list_history":
            print(f"[DEBUG] list_history 분기 실행")
            store = SQLStore()
            versions = store.get_document_versions(doc_id)
            if not versions:
                history = f"{doc_id} 문서의 버전을 찾을 수 없습니다."
            else:
                history = "\n".join([f"- v{v['version']} ({v['created_at']})" for v in versions])
            return {"context": [f"### [{doc_id} 버전 이력]\n{history} [DONE]"]}

        # [CASE 2] 버전 비교
        elif intent == "compare_versions":
            print(f"[DEBUG] compare_versions 분기 실행")
            # 버전 정보가 없으면 자동 추론 (최신 2개)
            if not v1 or not v2:
                print(f"[DEBUG] 버전 자동 선택 시작")
                store = SQLStore()
                versions = store.get_document_versions(doc_id)
                print(f"[DEBUG] 조회된 버전 개수: {len(versions) if versions else 0}")
                unique_versions = []
                seen = set()
                for v in versions:
                    if v['version'] not in seen:
                        unique_versions.append(v['version'])
                        seen.add(v['version'])
                
                print(f"[DEBUG] 고유 버전: {unique_versions}")
                
                if len(unique_versions) >= 2:
                    v1, v2 = unique_versions[1], unique_versions[0] # v1(이전), v2(최신)
                    print(f"     -> 자동 선택된 버전: {v1} vs {v2}")
                elif len(unique_versions) == 1:
                    return {"context": [f"### [{doc_id} 비교 불가]\n현재 문서의 버전이 하나({unique_versions[0]})뿐이라 비교할 대상이 없습니다. [DONE]"]}
                else:
                    return {"context": [f"### [{doc_id} 비교 불가]\n버전 목록을 가져올 수 없어 비교가 불가능합니다."]}

            # 실제 비교 데이터 조회 (SQL Diff) 및 검증
            print(f"[DEBUG] SQL Diff 조회 시작: {doc_id}, v1={v1}, v2={v2}")
            try:
                store = SQLStore()
                diffs = store.get_clause_diff(doc_id, v1, v2)

                # 디버그: 반환된 diff 확인
                print(f"[DEBUG] 총 diff 개수: {len(diffs)}")
                for d in diffs[:10]:  # 최대 10개만 출력
                    print(f"  - {d.get('clause', 'N/A')}: {d.get('change_type', 'N/A')}")
                
                # [Safety Check] "ADDED"로 표시된 항목이 v1.0 본문 어딘가에 숨어있는지 전수 조사
                added_items = [d for d in diffs if d['change_type'] == 'ADDED']
                if added_items:
                    # v1.0 문서의 모든 청크를 가져와서 하나의 텍스트로 합침
                    v1_doc = store.get_document_by_name(doc_id, v1)
                    if v1_doc:
                        v1_chunks = store.get_chunks_by_document(v1_doc['id'])
                        v1_full_text = "".join([c['content'] for c in v1_chunks]).replace(" ", "").replace("\n", "")
                        
                        for item in diffs:
                            if item['change_type'] == 'ADDED':
                                item_content_norm = (item['v2_content'] or "").replace(" ", "").replace("\n", "")
                                if len(item_content_norm) > 10 and item_content_norm in v1_full_text:
                                     item['change_type'] = 'UNCHANGED' 

                # 보고서용 데이터 포맷팅 (Strict Filter: MODIFIED ONLY)
                comp_lines = []
                for item in diffs:
                    if item['change_type'] == 'MODIFIED':
                        clause_id = item['clause'] or "N/A"
                        v1_txt = (item['v1_content'] or "").strip()
                        v2_txt = (item['v2_content'] or "").strip()
                        if v1_txt.replace(" ", "").replace("\n", "") != v2_txt.replace(" ", "").replace("\n", ""):
                            comp_lines.append(f"- [수정됨] 조항 {clause_id}: {v1_txt[:50]}... -> {v2_txt[:50]}...")

                comp_data = "\n".join(comp_lines)
                if not comp_data:
                    comp_data = "텍스트 내용이 변경된 조항이 감지되지 않았습니다. (단순 서식 변경이나 파싱 차이일 수 있습니다)."
                    
            except Exception as e:
                 print(f"[DEBUG compare.py] Diff 조회 실패: {e}")
                 return {"context": [f"### [비교 에이전트 오류]\n데이터 조회 중 오류가 발생했습니다: {str(e)}"]}

            # 영향 분석 조회 (Graph Impact) - 직접 GraphStore 사용
            try:
                graph_store = Neo4jGraphStore()
                graph_store.connect()
                impacts = graph_store.get_impact_analysis(doc_id)
                if not impacts:
                    impact_data = "이 문서의 변경으로 인해 영향을 받는 다른 문서는 발견되지 않았습니다."
                else:
                    impact_data = json.dumps(impacts, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"[DEBUG compare.py] Impact 조회 실패: {e}")
                impact_data = "영향 분석 데이터를 가져오는 중 오류가 발생했습니다."

            if not diffs: 
                 return {"context": [f"### [비교 에이전트 오류]\n{doc_id}의 지정된 버전({v1}, {v2}) 데이터를 가져올 수 없습니다. [DONE]"]}

            # 3. 종합 분석 (Z.AI)
            analysis_prompt = f"""다음은 두 버전의 문서 변경 사항(Diff)과, 해당 문서가 변경됨에 따라 영향을 받을 수 있는 다른 문서 목록(Impact)입니다.
            이를 종합하여 '팩트 기반의 변경 및 영향 분석 보고서'를 작성하세요.
            
            [1. 실제 텍스트 변경 조항 (MODIFIED Only)]
            {comp_data}
            
            [2. 영향 분석 (Impact Analysis)]
            {impact_data}
            
            [보고서 작성 절대 원칙 - 사용자 피드백 반영]
            1. **오직 위에 나열된 [1. 실제 텍스트 변경 조항]에 대해서만 설명하세요.**
               - 리스트에 없는 조항(예: 1.x, 2.x, 5.x 등)은 절대 언급하지 마세요.
            2. **마크다운(Markdown) 형식을 쓰지 마세요.**
               - '#', '**', '---' 같은 기호 없이 줄글과 들여쓰기만 사용하세요.
               - 깔끔한 텍스트 보고서 형식으로 작성하세요.
            3. **변경 내용을 구체적으로 비교하세요.**
               - "책임이 강화되었다" (X) -> "IT 관리자의 책임에 '로그 보존'이 추가되었습니다." (O)
            
            [보고서 형식]
            1. 변경 핵심 요약
               (바뀐 조항들만 간략히 언급)
            
            2. 상세 비교
               (조항별 변경 전/후 내용 비교)
               - 조항 4.1: ...
               - 조항 4.2: ...
            
            3. 영향 평가
               (변경된 조항과 관련된 영향 분석)
            """
            
            try:
                res = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": analysis_prompt}]
                )
                final_report = res.choices[0].message.content
                return {"context": [final_report + " [DONE]"]}
            except Exception as e:
                return {"context": [f"### [비교 보고서 생성 실패]\nLLM 호출 중 오류가 발생했습니다: {e} [DONE]"]}

        else:
             return {"context": [f"### [이해 불가]\n죄송합니다, 의도를 파악하지 못했습니다. (Intent: {intent}) [DONE]"]}
    
    except Exception as e:
        print(f"Compare Agent Error: {e}")
        return {"context": [f"### [에이전트 처리 오류]\n{str(e)} [DONE]"]}
