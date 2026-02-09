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
1. **정확성**: 반드시 제공된 [조사 결과]에 근거하여 답변하세요. 없는 내용을 지어내지 마세요.
2. **논리적 구성**: 여러 에이전트의 정보가 섞여 있다면, 이를 사용자가 이해하기 쉽게 논리적인 순서(예: 개요 -> 상세 절차 -> 관련 규정)로 재구성하세요.
3. **가독성**: 적절한 마크다운(불릿 포인트, 번호 매기기, 강조 등)을 사용하여 가독성을 높이세요.
4. **시각화 확인**: 조사 결과에 Mermaid 다이어그램 코드 등이 포함되어 있다면 이를 답변에 포함시켜 시각적 도움을 주세요.
5. **톤앤매너**: 전문적이면서도 친절한 한국어 어조를 유지하세요.
6. **마무리**: 답변 끝에 사용자가 추가로 궁금해할 만한 내용이 있다면 가볍게 언급해 줄 수 있습니다.

[주의사항]
- 서브 에이전트들의 시스템 태그나 내부 보고용 문구(예: [검색 에이전트 보고])는 답변에 그대로 노출하지 말고 자연스럽게 녹여내세요.
- 조사 결과가 부족하여 답변이 어렵다면, 아는 체하지 말고 "제공된 문서 내에서는 확인이 어렵다"고 솔직하게 답변하세요."""

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
