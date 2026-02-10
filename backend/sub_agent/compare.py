"""
비교 에이전트 (Comparison Agent) - Refactored Version
- 두 문서 또는 같은 문서의 서로 다른 버전 간 차이점을 분석하거나 버전 목록을 조회합니다.
- 결과를 context에 보고서 형태로 저장합니다.
"""

import json
from backend.agent import get_zai_client, get_version_history_tool, compare_versions_tool, get_impact_analysis_tool, AgentState

def comparison_agent_node(state: AgentState):
    """[서브] 비교 에이전트 - 버전 목록 조회 또는 내용 비교 분석"""
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
        info = json.loads(res.choices[0].message.content)
        intent = info.get("intent")
        doc_id = info.get("doc_id")
    except Exception as e:
        print(f"[DEBUG compare.py] 의도 분석 실패: {e}")
        return {"context": ["### [비교 에이전트 오류]\n사용자의 질문 의도를 파악하지 못했습니다. (예: SOP-001 버전 목록 보여줘 / SOP-001 v1, v2 비교해줘)"]}

    if not doc_id:
        return {"context": ["### [비교 에이전트 오류]\n분석할 문서 ID를 찾을 수 없습니다."]}

    # 2. 의도에 따른 도구 호출 및 처리
    if intent == "list_history":
        # 버전 목록 조회
        history = get_version_history_tool.invoke({"doc_id": doc_id})
        report = f"### [{doc_id} 버전 히스토리 목록]\n\n{history}\n\n[TIP] 특정 버전 두 개를 비교하고 싶으시면 'v1.0과 v2.0 비교해줘'라고 요청해 주세요. [DONE]"
        return {"context": [report]}
        
    elif intent == "compare_versions":
        # 내용 비교
        v1 = info.get("v1")
        v2 = info.get("v2")
        
        # 버전이 명시되지 않은 경우, 히스토리를 조회하여 최신 2개 버전을 자동으로 선택
        if not v1 or not v2:
            print(f"[DEBUG compare.py] 버전 미지정 -> 최신 2개 자동 선택 시도")
            history_str = get_version_history_tool.invoke({"doc_id": doc_id})
            
            # 히스토리 문자열에서 버전 번호 추출 (예: "v2.0", "1.0")
            import re
            versions = re.findall(r'v(\d+(?:\.\d+)*)', history_str)
            # 중복 제거 및 리스트 정렬 (버전이 1.0, 2.0 순서로 되어 있다고 가정하거나, DESC여도 순서는 상관없으나 최근 것이 중요)
            # 보통 SQLStore는 DESC로 반환하므로 순서대로 뽑으면 [v2.0, v1.0] 형태일 것임
            unique_versions = []
            for v in versions:
                if v not in unique_versions:
                    unique_versions.append(v)
            
            if len(unique_versions) >= 2:
                v1, v2 = unique_versions[1], unique_versions[0] # v1(이전), v2(최신)
                print(f"     -> 자동 선택된 버전: {v1} vs {v2}")
            elif len(unique_versions) == 1:
                return {"context": [f"### [{doc_id} 비교 불가]\n현재 문서의 버전이 하나({unique_versions[0]})뿐이라 비교할 대상이 없습니다. [DONE]"]}
            else:
                return {"context": [f"### [{doc_id} 비교 불가]\n버전 목록을 가져올 수 없어 비교가 불가능합니다."]}

        # 실제 비교 데이터 조회 (SQL Diff) 및 검증
        try:
            from backend.sql_store import SQLStore
            store = SQLStore()
            diffs = store.get_clause_diff(doc_id, v1, v2)
            
            # [Safety Check] "ADDED"로 표시된 항목이 v1.0 본문 어딘가에 숨어있는지 전수 조사
            added_items = [d for d in diffs if d['change_type'] == 'ADDED']
            if added_items:
                # v1.0 문서의 모든 청크를 가져와서 하나의 텍스트로 합침
                v1_doc = store.get_document_by_name(doc_id, v1)
                if v1_doc:
                    v1_chunks = store.get_chunks_by_document(v1_doc['id'])
                    # 공백 제거하여 정규화 (비교 정확도 향상)
                    v1_full_text = "".join([c['content'] for c in v1_chunks]).replace(" ", "").replace("\n", "")
                    
                    for item in diffs:
                        if item['change_type'] == 'ADDED':
                            # 해당 조항의 내용도 정규화
                            item_content_norm = (item['v2_content'] or "").replace(" ", "").replace("\n", "")
                            # v1 본문에 포함되어 있다면 'ADDED'가 아님 (내용이 10자 이상일 때만 체크)
                            if len(item_content_norm) > 10 and item_content_norm in v1_full_text:
                                 # 파싱 차이로 인한 것이므로 'UNCHANGED'로 변경하여 리포트 대상에서 제외
                                 item['change_type'] = 'UNCHANGED' 

            # [User Feedback Reflect] 
            # 사용자: "4.1~4.3만 바꿨는데 왜 딴게 나오냐" -> ADDED/DELETED는 파싱 노이즈일 가능성이 매우 높음
            # 따라서 'MODIFIED' 상태인 것만 리포트에 포함시킴.
            
            # 보고서용 데이터 포맷팅 (Strict Filter: MODIFIED ONLY)
            comp_lines = []
            for item in diffs:
                # 변경 유형이 MODIFIED인 경우만 포함 (내용 수정)
                if item['change_type'] == 'MODIFIED':
                    clause_id = item['clause'] or "N/A"
                    v1_txt = (item['v1_content'] or "").strip()
                    v2_txt = (item['v2_content'] or "").strip()
                    
                    # 텍스트 정규화 비교 (공백/줄바꿈 무시) 한 번 더 수행
                    if v1_txt.replace(" ", "").replace("\n", "") != v2_txt.replace(" ", "").replace("\n", ""):
                        comp_lines.append(f"- [수정됨] 조항 {clause_id}: {v1_txt[:50]}... -> {v2_txt[:50]}...")

            comp_data = "\n".join(comp_lines)
            if not comp_data:
                comp_data = "텍스트 내용이 변경된 조항이 감지되지 않았습니다. (단순 서식 변경이나 파싱 차이일 수 있습니다)."
                
        except Exception as e:
             print(f"[DEBUG compare.py] Diff 조회 실패: {e}")
             return {"context": [f"### [비교 에이전트 오류]\n데이터 조회 중 오류가 발생했습니다: {str(e)}"]}

        # 영향 분석 조회 (Graph Impact)
        impact_data = get_impact_analysis_tool.invoke({"doc_id": doc_id})

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
            analysis_res = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": analysis_prompt}]
            )
            # 보고서 본문에 버전 정보를 명시하여 답변 에이전트가 쉽게 인지하도록 함
            report_body = f"[{doc_id} 버전 비교 분석 결과]\n- 기준(이전) 버전: {v1}\n- 대상(최신) 버전: {v2}\n\n{analysis_res.choices[0].message.content}"
            report = f"### [{doc_id} v{v1} vs v{v2} 상세 비교 보고서]\n\n{report_body}\n\n[DONE]"
            return {"context": [report]}
        except Exception as e:
            return {"context": [f"### [비교 에이전트 분석 실패]\n분석 중 오류가 발생했습니다: {str(e)}"]}
    
    return {"context": ["### [비교 에이전트 알림]\n요청하신 작업을 수행할 수 없습니다."]}
