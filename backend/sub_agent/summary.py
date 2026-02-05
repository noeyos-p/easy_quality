import json
import re
from typing import Any, Dict, List, Optional
from backend.agent import get_zai_client, search_sop_tool, AgentState

def summary_agent_node(state: AgentState):
    """[서브] 요약 에이전트 (Z.AI) - 전체 요약 및 조항별 요약 지원"""
    client = get_zai_client()
    query = state["query"]
    model = state.get("model_name") or "glm-4.7-flash"
    
    # 1. 문서 검색 (기존 도구 활용)
    search_res = search_sop_tool.invoke({"query": query})
    
    # 2. 요약 모드 분석 (LLM 의도 파악)
    mode_prompt = f"""사용자의 질문 의도에 가장 적합한 요약 모드를 선택하세요.
    질문: {query}
    
    [모드 정의]
    - global_summary: 문서 전체의 핵심을 한눈에 파약하고 싶어할 때 (기본값)
    - section_summary: 문서를 조항별, 항목별, 구조별로 정리해달라는 명시적 의도가 있을 때
    
    반드시 JSON 형식으로만 답변하세요. 예: {{"mode": "global_summary"}}"""
    
    try:
        mode_res = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": mode_prompt}],
            response_format={"type": "json_object"}
        )
        mode = json.loads(mode_res.choices[0].message.content).get("mode", "global_summary")
    except:
        mode = "global_summary"

    # 3. 모드별 요약 수행
    if mode == "section_summary":
        system_content = """당신은 SOP 전문가입니다. 문서를 논리적 조항/섹션 구조에 따라 정리하여 **사용자에게 바로 전달될 최종 답변 형식**으로 작성하세요.
        - 형식: '1. 섹션명: 핵심 내용 요약'
        - 존재하는 섹션만 출력하고, 각 섹션은 2~5개의 짧은 불릿 포인트로 작성하세요.
        - 한국어로 친절하고 전문적인 말투로 답변하세요."""
        user_content = f"다음 문서를 조항별로 요약하세요.\n질문: {query}\n\n[문서 본문]\n{search_res}"
    else:
        system_content = """문서의 핵심을 6~10줄 또는 5~8개의 불릿 포인트로 요약하여 **사용자에게 바로 전달될 최종 답변 형식**으로 작성하세요. 
        핵심 위주로 간결하고 친절한 한국어로 답변하세요."""
        user_content = f"다음 문서를 요약하세요.\n질문: {query}\n\n[문서 본문]\n{search_res}"

    res = client.chat.completions.create(
        model=state.get("worker_model") or state.get("model_name") or "glm-4.7-flash", # 동적 모델 적용
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ]
    )
    
    report_tag = "[요약 에이전트 - 조항별 구조 정리]" if mode == "section_summary" else "[요약 에이전트 - 전체 핵심 요약]"
    final_answer = f"{report_tag}\n{res.choices[0].message.content}\n\n[DONE]"
    return {"messages": [{"role": "assistant", "content": final_answer}]}
