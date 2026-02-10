"""
최종 답변 생성 에이전트 (Answer Agent)
- 검색 에이전트가 제공한 [USE: ...] 태그를 수집하여 제거합니다.
- 마지막에 [참고 문서] 섹션을 자동 생성합니다.
"""

import re
from collections import defaultdict
from backend.agent import AgentState

def answer_agent_node(state: AgentState):
    """[서브] 답변 에이전트 - [USE: ...] 태그를 제거하고 [참고 문서] 섹션만 생성"""

    context_list = state.get("context", [])

    if not context_list:
        return {"messages": [{"role": "assistant", "content": "검색된 정보가 없습니다."}]}

    # context에서 검색 에이전트의 보고서 추출
    search_report = "\n\n".join(context_list)

    # "[검색 에이전트 조사 최종 보고]" 헤더 제거
    search_report = re.sub(r'###\s*\[검색 에이전트 조사 최종 보고\]\s*', '', search_report)

    # [DONE] 태그 제거 (나중에 [참고 문서] 뒤에 추가)
    search_report = re.sub(r'\s*\[DONE\]\s*$', '', search_report, flags=re.MULTILINE).strip()

    # [USE: ...] 태그 수집 (참고 문서 섹션 생성용)
    use_tags = re.findall(r'\[USE:\s*([^\|]+)\s*\|\s*([^\]]+)\]', search_report)
    doc_clauses = defaultdict(set)  # 문서별 조항 수집

    # [USE: 문서명 | 조항] 태그를 수집하고 제거
    def collect_and_remove_tag(match):
        """[USE: doc | clause] -> 태그 수집 후 제거 (조항 번호가 있는 것만)"""
        doc_name = match.group(1).strip()
        clause_info = match.group(2).strip()

        # 조항 정보에서 실제 조항 번호만 추출 (예: "5.1.3 제 3레벨(작업지침서(WI):" -> "5.1.3")
        # 조항 번호는 숫자.숫자 형식
        clause_match = re.match(r'([\d\.]+)', clause_info)
        if clause_match:
            clean_clause = clause_match.group(1)
            # 조항 번호가 유효한 경우만 수집 (숫자로 시작하고 최소 하나의 점이 있어야 함)
            if clean_clause and '.' in clean_clause and clean_clause[0].isdigit():
                doc_clauses[doc_name].add(clean_clause)
            else:
                print(f"[참고문서 필터링] 제외됨: {doc_name} > {clean_clause} (조항 형식 불일치)")
        else:
            print(f"[참고문서 필터링] 제외됨: {doc_name} > {clause_info} (조항 번호 없음)")

        # 인라인 인용 제거 - 빈 문자열 반환
        return ""

    # [USE: ...] 태그를 수집하고 제거 (인라인 인용 없이)
    converted = re.sub(
        r'\[USE:\s*([^\|]+)\s*\|\s*([^\]]+)\]',
        collect_and_remove_tag,
        search_report
    )

    # ========================================
    # [참고 문서] 섹션 자동 생성
    # ========================================
    if doc_clauses:
        ref_section = "\n\n[참고 문서]\n"
        for doc_name in sorted(doc_clauses.keys()):
            clauses = doc_clauses[doc_name]
            # 조항 번호 정렬
            try:
                sorted_clauses = sorted(clauses, key=lambda x: [int(n) if n.isdigit() else n for n in re.split(r'\.', x)])
            except:
                sorted_clauses = sorted(clauses)

            ref_section += f"{doc_name}({', '.join(sorted_clauses)})\n"

        converted += ref_section

    # [DONE] 태그를 마지막에 추가
    converted += "\n[DONE]"

    # ========================================
    # 검증 (Validation)
    # ========================================
    try:
        from backend.validation import validate_format, validate_coverage

        # 형식 검증
        format_result = validate_format(converted)
        if not format_result["valid"]:
            print(f"🔴 [답변 에이전트 검증 실패 - 형식]")
            for error in format_result["errors"]:
                print(f"   - {error}")

        # 커버리지 검증
        coverage_result = validate_coverage(state.get("query", ""), converted)
        if not coverage_result["valid"]:
            print(f"🔴 [답변 에이전트 검증 경고 - 커버리지]")
            for warning in coverage_result["warnings"]:
                print(f"   - {warning}")

    except Exception as e:
        print(f"🔴 [검증 모듈 로드 실패] {e}")

    return {"messages": [{"role": "assistant", "content": converted}]}
