"""
최종 답변 생성 에이전트 (Answer Agent)
- 서브 에이전트들이 수집하여 context에 쌓아둔 정보를 기반으로 사용자에게 전달할 최종 답변을 작성합니다.
"""

from backend.agent import get_zai_client, AgentState

def answer_agent_node(state: AgentState):
    """[서브] 답변 에이전트 - 수집된 모든 context를 종합하여 최종 답변 생성"""
    client = get_zai_client()
    query = state["query"]
    # 여러 서브 에이전트가 context를 리스트 형태로 쌓았을 경우를 대비 (Annotated[List[str], operator.add])
    context = state.get("context", "")
    
    if isinstance(context, list):
        combined_context = "\n\n---\n\n".join(context)
    else:
        combined_context = context

    model = state.get("worker_model") or state.get("model_name") or "glm-4.7-flash"

    system_prompt = """당신은 GMP 규정 전문가이자 전문 작가입니다. 
하위 전문가 에이전트들이 조사하여 보고한 [조사 결과]를 바탕으로 사용자의 [질문]에 대해 가장 적절하고 친절한 최종 답변을 작성하세요.

[답변 가이드라인]
1. **STRICT GROUNDING (최우선)**: 당신은 오직 제공된 [조사 결과] 내의 텍스트만을 답변의 유일한 재료로 사용해야 합니다.
2. **NO EXTERNAL KNOWLEDGE**: 당신이 평소에 알고 있는 GMP 상식, 일반적인 절차, 혹은 외부 지식을 답변에 조금이라도 섞는 것은 **심각한 결함**입니다.
3. **증명 가능성**: 답변의 모든 문장은 [조사 결과] 내의 특정 구문으로 직접 증명 가능해야 합니다.
4. **논리적 구성**: 여러 에이전트의 정보가 섞여 있다면, 이를 사용자가 이해하기 쉽게 [조사 결과] 내 범위에서만 재구성하세요.
5. **가독성**: 적절한 마크다운을 사용하되, 의미를 왜곡하지 마세요.
6. **부재 시 대응**: 만약 질문에 대한 답이 [조사 결과]에 없다면, "제공된 문서 및 조사 결과 내에서는 관련 내용을 확인할 수 없습니다."라고만 답변하세요. 절대 추측하거나 일반론을 말하지 마세요.

[주의사항]
- 서브 에이전트들의 시스템 태그(예: [검색 에이전트 보고])는 답변에서 자연스럽게 제거하세요.
- 당신의 역할은 '창의적 작가'가 아니라 '조사 결과의 성실한 대변인'입니다.
"""

    user_content = f"사용자 질문: {query}\n\n[조사 결과]\n{combined_context}"

    res = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ],
        temperature=0.3 # 작문이므로 약간의 유연성 부여
    )

    final_answer = res.choices[0].message.content
    
    # 마지막 응답임을 나타내는 메시지 반환
    return {"messages": [{"role": "assistant", "content": final_answer}]}
