"""
최종 답변 생성 에이전트 (Answer Agent)
- 검색 에이전트가 제공한 [USE: ...] 태그를 (문서명 > 조항) 형식으로 정확하게 변환합니다.
"""

import re
from backend.agent import AgentState

def answer_agent_node(state: AgentState):
    """[서브] 답변 에이전트 - [USE: ...] 태그를 (문서명 > 조항) 형식으로 변환"""

    context_list = state.get("context", [])

    if not context_list:
        return {"messages": [{"role": "assistant", "content": "검색된 정보가 없습니다."}]}

    # context에서 검색 에이전트의 보고서 추출
    search_report = "\n\n".join(context_list)

    # "[검색 에이전트 조사 최종 보고]" 헤더 제거
    search_report = re.sub(r'###\s*\[검색 에이전트 조사 최종 보고\]\s*', '', search_report)

    # [USE: 문서명 | 조항] 태그를 (문서명 > 조항) 형식으로 정확하게 변환
    def replace_use_tag(match):
        """[USE: doc | clause] -> (doc > clause) 정확한 변환"""
        doc_name = match.group(1).strip()
        clause_info = match.group(2).strip()

        # 조항 정보에서 실제 조항 번호만 추출 (예: "5.1.3 제 3레벨(작업지침서(WI):" -> "5.1.3")
        # 조항 번호는 숫자.숫자 형식
        clause_match = re.match(r'([\d\.]+)', clause_info)
        if clause_match:
            clean_clause = clause_match.group(1)
        else:
            clean_clause = clause_info

        return f" ({doc_name} > {clean_clause})"

    # [USE: ...] 태그를 정규식으로 정확하게 변환
    converted = re.sub(
        r'\[USE:\s*([^\|]+)\s*\|\s*([^\]]+)\]',
        replace_use_tag,
        search_report
    )

    return {"messages": [{"role": "assistant", "content": converted}]}
