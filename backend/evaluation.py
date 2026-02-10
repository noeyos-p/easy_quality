"""
LLM as a Judge - 에이전트 응답 평가 모듈

평가 메트릭:
- faithfulness: 답변이 제공된 컨텍스트에 충실한지 (환각 방지)
- groundness: 답변이 실제 근거에 기반하고 있는지
- relevancy: 답변이 질문과 관련이 있는지
- correctness: 답변이 정확하고 완전한지

특징:
- **무조건 RDB(PostgreSQL)에서 실제 문서를 조회하여 인용 정확성 검증**
- [참고 문서] 섹션의 조항 번호가 실제로 존재하는지 DB에서 확인
- 인라인 인용 (문서명 > 조항)도 검증
"""

import os
import json
import re
from typing import Dict, List, Optional, Tuple
from openai import OpenAI


class AgentEvaluator:
    """LLM as a Judge를 사용한 에이전트 응답 평가기 (RDB 검증 필수)"""

    def __init__(self, judge_model: str = "gpt-4o-mini", sql_store=None):
        """
        Args:
            judge_model: 평가에 사용할 LLM 모델
            sql_store: SQL 데이터베이스 스토어 (실제 문서 검증용 - 필수!)
        """
        if not sql_store:
            raise ValueError("sql_store는 필수입니다. RDB에서 문서를 검증해야 합니다.")

        self.judge_model = judge_model
        self.sql_store = sql_store

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")

        self.client = OpenAI(api_key=api_key)

    def _call_judge(self, prompt: str) -> Dict:
        """Judge LLM 호출"""
        try:
            response = self.client.chat.completions.create(
                model=self.judge_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"🔴 Judge 호출 실패: {e}")
            return {"score": 0, "reasoning": f"평가 실패: {str(e)}"}

    def _extract_all_citations(self, answer: str) -> List[Tuple[str, str]]:
        """
        답변에서 모든 인용 추출 (RDB 검증용)

        Returns:
            [(doc_id, clause_num), ...]
            예: [("EQ-SOP-00001", "5.1.3"), ("EQ-SOP-00002", "3.2")]
        """
        citations = []

        # 1. [참고 문서] 섹션: EQ-SOP-00001(5.1.3, 5.4.2)
        ref_pattern = r'\[참고 문서\](.*?)(?:\[DONE\]|$)'
        ref_match = re.search(ref_pattern, answer, re.DOTALL)

        if ref_match:
            ref_section = ref_match.group(1).strip()
            # 문서 ID와 조항 파싱
            doc_pattern = r'([A-Z]+-[A-Z]+-\d+)\s*\(([^)]+)\)'
            for match in re.finditer(doc_pattern, ref_section):
                doc_id = match.group(1)
                clauses_str = match.group(2)
                clauses = [c.strip() for c in clauses_str.split(',')]
                for clause in clauses:
                    # 조항 번호만 추출 (숫자.숫자 형식)
                    clause_match = re.match(r'([\d\.]+)', clause)
                    if clause_match:
                        citations.append((doc_id, clause_match.group(1)))

        # 2. 인라인 인용: (EQ-SOP-00001 > 3.1)
        inline_pattern = r'\(([A-Z]+-[A-Z]+-\d+)\s*>\s*([\d\.]+)'
        for match in re.finditer(inline_pattern, answer):
            doc_id = match.group(1)
            clause_num = match.group(2)
            citations.append((doc_id, clause_num))

        # 3. 마크다운 강조 인용: **[EQ-SOP-00001 > 3.1]**
        markdown_pattern = r'\*\*\[([A-Z]+-[A-Z]+-\d+)\s*>\s*([\d\.]+)'
        for match in re.finditer(markdown_pattern, answer):
            doc_id = match.group(1)
            clause_num = match.group(2)
            citations.append((doc_id, clause_num))

        # 중복 제거 (순서 유지)
        seen = set()
        unique_citations = []
        for doc_id, clause_num in citations:
            key = (doc_id, clause_num)
            if key not in seen:
                seen.add(key)
                unique_citations.append((doc_id, clause_num))

        return unique_citations

    def _verify_against_rdb(self, answer: str) -> Dict:
        """
        **핵심: RDB에서 실제 문서를 조회하여 답변 검증**

        Returns:
            {
                "has_citations": bool,
                "total_citations": int,
                "verified_citations": int,
                "incorrect_citations": List[str],
                "accuracy_rate": float,
                "verification_details": str
            }
        """
        # 답변에서 모든 인용 추출
        citations = self._extract_all_citations(answer)

        if not citations:
            return {
                "has_citations": False,
                "total_citations": 0,
                "verified_citations": 0,
                "incorrect_citations": [],
                "accuracy_rate": 0.0,
                "verification_details": "❌ 인용 없음 - 답변에 문서 근거가 명시되지 않음"
            }

        total_citations = len(citations)
        verified_citations = 0
        incorrect_citations = []
        details = []

        # 각 인용을 RDB에서 검증
        for doc_id, clause_num in citations:
            try:
                # RDB에서 문서 조회 (최신 버전)
                doc = self.sql_store.get_document_by_name(doc_id)

                if not doc:
                    incorrect_citations.append(f"{doc_id}:{clause_num}")
                    details.append(f"❌ {doc_id}:{clause_num} - 문서가 RDB에 없음")
                    continue

                # 문서의 청크 조회
                chunks = self.sql_store.get_chunks_by_document(doc['id'])

                # 해당 조항 번호의 청크 찾기
                found = False
                for chunk in chunks:
                    chunk_clause = chunk.get('clause', '')
                    # 조항 번호 정규화 비교 (7.5 == 7.5.0)
                    if chunk_clause == clause_num or chunk_clause.startswith(clause_num + '.'):
                        found = True
                        break

                if found:
                    verified_citations += 1
                    details.append(f"✅ {doc_id}:{clause_num}")
                else:
                    incorrect_citations.append(f"{doc_id}:{clause_num}")
                    details.append(f"❌ {doc_id}:{clause_num} - 조항이 문서에 없음")

            except Exception as e:
                incorrect_citations.append(f"{doc_id}:{clause_num}")
                details.append(f"⚠️ {doc_id}:{clause_num} - RDB 조회 오류: {str(e)}")

        accuracy_rate = (verified_citations / total_citations * 100) if total_citations > 0 else 0.0

        return {
            "has_citations": True,
            "total_citations": total_citations,
            "verified_citations": verified_citations,
            "incorrect_citations": incorrect_citations,
            "accuracy_rate": round(accuracy_rate, 1),
            "verification_details": "\\n".join(details)
        }

    def evaluate_faithfulness(self, question: str, answer: str, context: str) -> Dict:
        """
        충실성 평가: 답변이 제공된 컨텍스트에만 기반하는지 평가
        **RDB 검증 결과를 최우선으로 고려**
        """
        # RDB 검증 (필수)
        rdb_verification = self._verify_against_rdb(answer)

        rdb_info = f"""
[RDB 검증 결과 - 최우선 고려!]
- 인용 존재: {'예' if rdb_verification['has_citations'] else '아니오'}
- 검증된 인용: {rdb_verification['verified_citations']}/{rdb_verification['total_citations']}
- 정확도: {rdb_verification['accuracy_rate']}%
- 틀린 인용: {len(rdb_verification['incorrect_citations'])}개

검증 상세:
{rdb_verification['verification_details']}
"""

        prompt = f"""답변이 컨텍스트에만 충실하게 기반하고 있는지 평가하세요.

**RDB 검증 결과를 최우선으로 고려하세요!**
{rdb_info}

[질문] {question}
[컨텍스트] {context[:2000]}...
[답변] {answer[:1500]}...

[평가 기준 - RDB 검증 결과 중심]
5점: RDB 정확도 100%, 모든 인용이 실제 문서에 존재, 컨텍스트만 사용
4점: RDB 정확도 90% 이상, 1개 인용만 불일치
3점: RDB 정확도 70-89%, 2-3개 인용 불일치
2점: RDB 정확도 50-69%, 30% 이상 불일치
1점: RDB 정확도 50% 미만 또는 인용 없음

JSON 형식으로 답변:
{{"score": 1-5, "reasoning": "RDB 검증 결과 중심으로 평가 근거 설명"}}"""

        result = self._call_judge(prompt)
        result["rdb_verification"] = rdb_verification
        return result

    def evaluate_groundness(self, question: str, answer: str, context: str) -> Dict:
        """
        근거성 평가: 답변의 모든 주장이 명확한 근거를 가지는지
        **RDB 검증 필수**
        """
        rdb_verification = self._verify_against_rdb(answer)

        rdb_info = f"""
[RDB 검증 결과 - 최우선!]
- 인용 존재: {'예' if rdb_verification['has_citations'] else '아니오'}
- 검증된 인용: {rdb_verification['verified_citations']}/{rdb_verification['total_citations']}
- 정확도: {rdb_verification['accuracy_rate']}%
"""

        prompt = f"""답변의 근거성을 평가하세요.

**RDB 검증 결과를 최우선으로 고려하세요!**
{rdb_info}

[질문] {question}
[답변] {answer[:1500]}...

[평가 기준]
5점: 인용 있고 RDB 정확도 100%, 모든 주장에 명확한 근거
4점: 인용 있고 RDB 정확도 90% 이상
3점: 인용 있고 RDB 정확도 70-89%
2점: 인용 있고 RDB 정확도 50-69%
1점: 인용 없음 또는 RDB 정확도 50% 미만

JSON 형식으로 답변:
{{"score": 1-5, "reasoning": "RDB 검증 결과 중심으로 평가"}}"""

        result = self._call_judge(prompt)
        result["rdb_verification"] = rdb_verification
        return result

    def evaluate_relevancy(self, question: str, answer: str, context: str = None) -> Dict:
        """관련성 평가: 답변이 질문에 직접적으로 답하는지"""
        prompt = f"""답변이 질문에 직접적으로 답하는지 평가하세요.

[질문] {question}
[답변] {answer[:1500]}...

[평가 기준]
5점: 질문의 핵심을 정확히 파악하고 직접 답변, 불필요한 정보 없음
4점: 질문에 잘 답하나 일부 부가 정보 포함
3점: 질문과 관련있으나 우회적이거나 일부 무관한 내용
2점: 질문과 부분적으로만 관련, 주요 내용 누락
1점: 질문과 거의 무관

JSON 형식으로 답변:
{{"score": 1-5, "reasoning": "평가 근거"}}"""

        return self._call_judge(prompt)

    def evaluate_correctness(
        self,
        question: str,
        answer: str,
        context: str,
        reference_answer: str = None
    ) -> Dict:
        """
        정확성 평가: 답변이 정확하고 완전한지
        **RDB 검증 필수**
        """
        rdb_verification = self._verify_against_rdb(answer)

        rdb_info = f"""
[RDB 검증 결과 - 최우선!]
- 인용 존재: {'예' if rdb_verification['has_citations'] else '아니오'}
- 검증된 인용: {rdb_verification['verified_citations']}/{rdb_verification['total_citations']}
- 정확도: {rdb_verification['accuracy_rate']}%
- 틀린 인용: {', '.join(rdb_verification['incorrect_citations'][:5])}
"""

        ref_section = f"\\n[참조 답변] {reference_answer}\\n" if reference_answer else ""

        prompt = f"""답변이 정확하고 완전한지 평가하세요.

**RDB 검증 결과를 최우선으로 고려하세요!**
{rdb_info}

[질문] {question}
[컨텍스트] {context[:2000]}...
{ref_section}
[답변] {answer[:1500]}...

[평가 기준 - RDB 검증 중심]
5점: RDB 정확도 100%, 모든 내용 정확, 필요한 정보 모두 포함
4점: RDB 정확도 90% 이상, 대체로 정확하나 일부 세부사항 누락
3점: RDB 정확도 70-89%, 핵심은 맞으나 중요 정보 누락
2점: RDB 정확도 50-69%, 여러 오류 존재
1점: RDB 정확도 50% 미만, 인용 없음, 심각한 오류 다수

JSON 형식으로 답변:
{{"score": 1-5, "reasoning": "RDB 검증 결과 중심으로 평가. 틀린 인용 있으면 구체적으로 지적"}}"""

        result = self._call_judge(prompt)
        result["rdb_verification"] = rdb_verification
        return result

    def evaluate_single(
        self,
        question: str,
        answer: str,
        context: str = "",
        metrics: List[str] = None,
        reference_answer: str = None
    ) -> Dict[str, Dict]:
        """
        단일 질문-답변 쌍에 대한 종합 평가

        Args:
            question: 사용자 질문
            answer: 에이전트 답변
            context: 검색된 컨텍스트
            metrics: 평가할 메트릭 리스트 (기본: 전체)
            reference_answer: 참조 답변 (옵션)

        Returns:
            {"faithfulness": {...}, "groundness": {...}, "relevancy": {...}, "correctness": {...}}
        """
        if metrics is None:
            metrics = ["faithfulness", "groundness", "relevancy", "correctness"]

        results = {}

        print("\\n" + "="*80)
        print("🔍 LLM as a Judge 평가 시작 (RDB 검증 포함)")
        print("="*80)

        if "faithfulness" in metrics and context:
            print("\\n[1/4] Faithfulness 평가 중...")
            results["faithfulness"] = self.evaluate_faithfulness(question, answer, context)
            print(f"   점수: {results['faithfulness'].get('score', 0)}/5")

        if "groundness" in metrics and context:
            print("\\n[2/4] Groundness 평가 중...")
            results["groundness"] = self.evaluate_groundness(question, answer, context)
            print(f"   점수: {results['groundness'].get('score', 0)}/5")

        if "relevancy" in metrics:
            print("\\n[3/4] Relevancy 평가 중...")
            results["relevancy"] = self.evaluate_relevancy(question, answer, context)
            print(f"   점수: {results['relevancy'].get('score', 0)}/5")

        if "correctness" in metrics and context:
            print("\\n[4/4] Correctness 평가 중...")
            results["correctness"] = self.evaluate_correctness(question, answer, context, reference_answer)
            print(f"   점수: {results['correctness'].get('score', 0)}/5")

        # 평균 점수 계산
        scores = [r.get("score", 0) for r in results.values() if "score" in r]
        avg_score = sum(scores) / len(scores) if scores else 0

        print("\\n" + "="*80)
        print(f"📊 평가 완료 - 평균 점수: {avg_score:.1f}/5")
        print("="*80)

        results["average_score"] = round(avg_score, 1)
        return results
