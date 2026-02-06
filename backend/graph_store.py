"""
Neo4j 그래프 저장소

노드 타입:
- Document: 문서 관리
- Section: 조항 관리 + LLM 메타데이터
- DocumentType: 문서 유형 (SOP, WI, FORM 등)
- Concept: 관리 영역 (user_account, document_lifecycle, training 등)
- Question: RAG 질문 추적

관계 타입:
- HAS_SECTION: Document -> Section
- PARENT_OF: Section -> Section (계층)
- REFERENCES: Document -> Document (문서 간 참조)
- IS_TYPE: Document -> DocumentType
- MENTIONS: Section -> Document (조항 내 타 문서 언급)
- BELONGS_TO_CONCEPT: Section -> Concept
- USED_SECTION: Question -> Section (RAG 추적)
"""

from neo4j import GraphDatabase
from typing import List, Dict, Optional
import re
import uuid


class Neo4jGraphStore:
    """Neo4j 그래프 저장소"""

    def __init__(
        self,
        uri: str = "neo4j+s://d00efa60.databases.neo4j.io",
        user: str = "neo4j",
        password: str = "4Qs45al1Coz_NwZDSMcFV9JIFjU7zXPjdKyptQloS6c",
        database: str = "neo4j"
    ):
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database
        self.driver = None

    def connect(self):
        """Neo4j 연결"""
        if not self.driver:
            try:
                self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
                if self.test_connection():
                    print(f"🟢 Neo4j 연결 성공")
            except Exception as e:
                print(f"🔴 Neo4j 연결 실패: {e}")
        return self

    def close(self):
        if self.driver:
            self.driver.close()
            self.driver = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def test_connection(self) -> bool:
        try:
            with self.driver.session(database=self.database) as session:
                session.run("RETURN 1")
                return True
        except:
            return False

    # ═══════════════════════════════════════════════════════════════════════════
    # 스키마 초기화
    # ═══════════════════════════════════════════════════════════════════════════

    def init_schema(self):
        """인덱스 및 제약조건"""
        constraints = [
            "CREATE CONSTRAINT doc_id IF NOT EXISTS FOR (d:Document) REQUIRE d.doc_id IS UNIQUE",
            "CREATE CONSTRAINT section_id IF NOT EXISTS FOR (s:Section) REQUIRE s.section_id IS UNIQUE",
            "CREATE CONSTRAINT question_id IF NOT EXISTS FOR (q:Question) REQUIRE q.id IS UNIQUE",
            "CREATE CONSTRAINT doc_type_code IF NOT EXISTS FOR (dt:DocumentType) REQUIRE dt.code IS UNIQUE",
            "CREATE CONSTRAINT concept_id IF NOT EXISTS FOR (c:Concept) REQUIRE c.concept_id IS UNIQUE",
            "CREATE INDEX doc_title IF NOT EXISTS FOR (d:Document) ON (d.title)",
            "CREATE INDEX section_title IF NOT EXISTS FOR (s:Section) ON (s.title)",
            "CREATE INDEX section_intent_scope IF NOT EXISTS FOR (s:Section) ON (s.intent_scope)",
        ]

        with self.driver.session(database=self.database) as session:
            for c in constraints:
                try:
                    session.run(c)
                except:
                    pass
        print("🟢 스키마 초기화 완료")

    def clear_all(self):
        """모든 데이터 삭제"""
        with self.driver.session(database=self.database) as session:
            session.run("MATCH (n) DETACH DELETE n")
        print("🟢 모든 데이터 삭제 완료")

    # ═══════════════════════════════════════════════════════════════════════════
    # Document 관리
    # ═══════════════════════════════════════════════════════════════════════════

    def create_document(self, doc_id: str, title: str, version: str = "1.0",
                       effective_date: str = "", owning_dept: str = "", **metadata):
        """문서 생성"""
        with self.driver.session(database=self.database) as session:
            session.run("""
                MERGE (d:Document {doc_id: $doc_id})
                SET d.title = $title,
                    d.version = $version,
                    d.effective_date = $effective_date,
                    d.owning_dept = $owning_dept,
                    d.updated_at = datetime()
            """, doc_id=doc_id, title=title, version=version,
                effective_date=effective_date, owning_dept=owning_dept)

    def create_document_type(self, code: str, name_kr: str, name_en: str):
        """DocumentType 노드 생성"""
        with self.driver.session(database=self.database) as session:
            session.run("""
                MERGE (dt:DocumentType {code: $code})
                SET dt.name_kr = $name_kr, dt.name_en = $name_en
            """, code=code, name_kr=name_kr, name_en=name_en)

    def link_document_type(self, doc_id: str, type_code: str):
        """Document -[:IS_TYPE]-> DocumentType 관계"""
        with self.driver.session(database=self.database) as session:
            session.run("""
                MATCH (d:Document {doc_id: $doc_id})
                MATCH (dt:DocumentType {code: $type_code})
                MERGE (d)-[:IS_TYPE]->(dt)
            """, doc_id=doc_id, type_code=type_code)

    def create_concept(self, concept_id: str, name_kr: str, name_en: str, description: str = ""):
        """Concept 노드 생성 (관리 영역)"""
        with self.driver.session(database=self.database) as session:
            session.run("""
                MERGE (c:Concept {concept_id: $concept_id})
                SET c.name_kr = $name_kr,
                    c.name_en = $name_en,
                    c.description = $description
            """, concept_id=concept_id, name_kr=name_kr, name_en=name_en, description=description)

    def link_section_concept(self, section_id: str, concept_id: str):
        """Section -[:BELONGS_TO_CONCEPT]-> Concept 관계"""
        with self.driver.session(database=self.database) as session:
            session.run("""
                MATCH (s:Section {section_id: $section_id})
                MATCH (c:Concept {concept_id: $concept_id})
                MERGE (s)-[:BELONGS_TO_CONCEPT]->(c)
            """, section_id=section_id, concept_id=concept_id)

    def get_document(self, doc_id: str) -> Optional[Dict]:
        """문서 조회"""
        with self.driver.session(database=self.database) as session:
            result = session.run("""
                MATCH (d:Document {doc_id: $doc_id})
                OPTIONAL MATCH (d)-[:HAS_SECTION]->(s:Section)
                RETURN d, count(s) as section_count
            """, doc_id=doc_id)
            record = result.single()
            if record:
                return {**dict(record["d"]), "section_count": record["section_count"]}
            return None

    def get_all_documents(self) -> List[Dict]:
        """모든 문서 목록"""
        with self.driver.session(database=self.database) as session:
            result = session.run("""
                MATCH (d:Document)
                OPTIONAL MATCH (d)-[:HAS_SECTION]->(s:Section)
                RETURN d, count(s) as section_count
                ORDER BY d.doc_id
            """)
            return [{**dict(r["d"]), "section_count": r["section_count"]} for r in result]

    def delete_document(self, doc_id: str):
        """문서 및 관련 섹션 삭제"""
        with self.driver.session(database=self.database) as session:
            session.run("""
                MATCH (d:Document {doc_id: $doc_id})
                OPTIONAL MATCH (d)-[:HAS_SECTION]->(s:Section)
                DETACH DELETE d, s
            """, doc_id=doc_id)
        print(f"🟢 문서 삭제: {doc_id}")

    # ═══════════════════════════════════════════════════════════════════════════
    # Section 관리
    # ═══════════════════════════════════════════════════════════════════════════

    def create_section(self, doc_id: str, section_id: str, title: str, content: str,
                      clause_level: int = 0, main_section: str = None, llm_meta: Dict = None, **kwargs):
        """섹션 생성 + LLM 메타데이터 (evaluate_gmp_unified 호환)"""
        meta = llm_meta or {}

        # main_section 기본값
        if not main_section:
            main_section = section_id.split('.')[0] if '.' in section_id else section_id

        with self.driver.session(database=self.database) as session:
            session.run("""
                MATCH (d:Document {doc_id: $doc_id})
                MERGE (s:Section {section_id: $section_id})
                SET s.doc_id = $doc_id,
                    s.title = $title,
                    s.content = $content,
                    s.clause_level = $clause_level,
                    s.main_section = $main_section,
                    s.content_type = $content_type,
                    s.main_topic = $main_topic,
                    s.sub_topics = $sub_topics,
                    s.actors = $actors,
                    s.actions = $actions,
                    s.conditions = $conditions,
                    s.summary = $summary,
                    s.intent_scope = $intent_scope,
                    s.intent_summary = $intent_summary,
                    s.language = $language
                MERGE (d)-[:HAS_SECTION]->(s)
            """,
            doc_id=doc_id,
            section_id=section_id,
            title=title,
            content=content,
            clause_level=clause_level,
            main_section=main_section,
            content_type=meta.get("content_type", ""),
            main_topic=meta.get("main_topic", ""),
            sub_topics=str(meta.get("sub_topics", [])),
            actors=str(meta.get("actors", [])),
            actions=str(meta.get("actions", [])),
            conditions=str(meta.get("conditions", [])),
            summary=meta.get("summary", ""),
            intent_scope=meta.get("intent_scope", ""),
            intent_summary=meta.get("intent_summary", ""),
            language=meta.get("language", "ko")
            )

    def create_section_hierarchy(self, parent_id: str, child_id: str):
        """섹션 계층 관계 (같은 문서 내에서만)"""
        with self.driver.session(database=self.database) as session:
            session.run("""
                MATCH (p:Section {section_id: $parent})
                MATCH (c:Section {section_id: $child})
                WHERE p.doc_id = c.doc_id
                MERGE (p)-[:PARENT_OF]->(c)
            """, parent=parent_id, child=child_id)

    def get_section_hierarchy(self, doc_id: str) -> List[Dict]:
        """문서의 섹션 계층"""
        with self.driver.session(database=self.database) as session:
            result = session.run("""
                MATCH (d:Document {doc_id: $doc_id})-[:HAS_SECTION]->(s:Section)
                OPTIONAL MATCH (s)-[:PARENT_OF]->(child:Section)
                RETURN s, collect(child.section_id) as children
                ORDER BY s.section_id
            """, doc_id=doc_id)
            return [{"section": dict(r["s"]), "children": r["children"]} for r in result]

    def search_sections(self, keyword: str, doc_id: str = None) -> List[Dict]:
        """섹션 검색"""
        query = """
            MATCH (s:Section)
            WHERE toLower(s.content) CONTAINS toLower($keyword)
               OR toLower(s.title) CONTAINS toLower($keyword)
        """
        if doc_id:
            query += " AND s.doc_id = $doc_id"
        query += " RETURN s LIMIT 10"

        with self.driver.session(database=self.database) as session:
            result = session.run(query, keyword=keyword, doc_id=doc_id)
            return [dict(r["s"]) for r in result]

    # ═══════════════════════════════════════════════════════════════════════════
    # 문서 간 참조
    # ═══════════════════════════════════════════════════════════════════════════

    def create_reference(self, from_doc: str, to_doc: str):
        """문서 간 참조 관계"""
        with self.driver.session(database=self.database) as session:
            session.run("""
                MATCH (from:Document {doc_id: $from})
                MATCH (to:Document {doc_id: $to})
                MERGE (from)-[:REFERENCES]->(to)
            """, from_doc=from_doc, to=to_doc)

    def link_section_mentions(self, section_id: str, mentioned_docs: List[str]):
        """섹션에서 언급한 문서들 연결 (Section -[:MENTIONS]-> Document)"""
        with self.driver.session(database=self.database) as session:
            for doc_id in mentioned_docs:
                # MERGE: 참조된 Document가 없으면 자동 생성
                session.run("""
                    MATCH (s:Section {section_id: $section})
                    MERGE (d:Document {doc_id: $doc})
                    ON CREATE SET d.title = $doc, d.version = "", d.effective_date = "", d.owning_dept = ""
                    MERGE (s)-[:MENTIONS]->(d)
                """, section=section_id, doc=doc_id)

    def get_document_references(self, doc_id: str) -> Dict:
        """문서 참조 관계 (MENTIONS 기반)"""
        with self.driver.session(database=self.database) as session:
            result = session.run("""
                MATCH (d:Document {doc_id: $doc_id})

                // 이 문서의 섹션들이 MENTIONS하는 문서들
                OPTIONAL MATCH (d)-[:HAS_SECTION]->(s:Section)-[:MENTIONS]->(ref:Document)

                // 다른 문서의 섹션들이 이 문서를 MENTIONS하는 경우
                OPTIONAL MATCH (citing_section:Section)-[:MENTIONS]->(d)
                OPTIONAL MATCH (citing_doc:Document)-[:HAS_SECTION]->(citing_section)

                RETURN d,
                       collect(DISTINCT ref.doc_id) as references,
                       collect(DISTINCT citing_doc.doc_id) as cited_by
            """, doc_id=doc_id)
            record = result.single()
            if record:
                # null 값 제거
                references = [r for r in record["references"] if r]
                cited_by = [c for c in record["cited_by"] if c]
                return {
                    "document": dict(record["d"]),
                    "references": references,
                    "cited_by": cited_by
                }
            return None

    # ═══════════════════════════════════════════════════════════════════════════
    # Question 추적 (RAG 설명 가능성)
    # ═══════════════════════════════════════════════════════════════════════════

    def create_question(self, question_id: str, text: str, answer: str = None,
                       session_id: str = None, llm_model: str = None):
        """질문 기록"""
        with self.driver.session(database=self.database) as session:
            session.run("""
                MERGE (q:Question {id: $id})
                SET q.text = $text,
                    q.answer = $answer,
                    q.session_id = $session_id,
                    q.llm_model = $llm_model,
                    q.created_at = datetime()
            """, id=question_id, text=text, answer=answer, session_id=session_id, llm_model=llm_model)

    def link_question_to_section(self, question_id: str, section_id: str, rank: int, score: float):
        """질문이 참조한 섹션 연결"""
        with self.driver.session(database=self.database) as session:
            session.run("""
                MATCH (q:Question {id: $q_id})
                MATCH (s:Section {section_id: $s_id})
                MERGE (q)-[r:USED_SECTION]->(s)
                SET r.rank = $rank, r.score = $score
            """, q_id=question_id, s_id=section_id, rank=rank, score=score)

    def get_question_sources(self, question_id: str) -> Dict:
        """질문이 참조한 섹션 조회"""
        with self.driver.session(database=self.database) as session:
            result = session.run("""
                MATCH (q:Question {id: $id})
                OPTIONAL MATCH (q)-[r:USED_SECTION]->(s:Section)
                RETURN q, collect({section: s, rank: r.rank, score: r.score}) as sources
                ORDER BY r.rank
            """, id=question_id)
            record = result.single()
            if record:
                return {
                    "question": dict(record["q"]),
                    "sources": [
                        {
                            "section": dict(s["section"]) if s["section"] else None,
                            "rank": s["rank"],
                            "score": s["score"]
                        }
                        for s in record["sources"] if s["section"]
                    ]
                }
            return None

    def get_question_history(self, session_id: str = None, limit: int = 50) -> List[Dict]:
        """질문 히스토리"""
        query = "MATCH (q:Question)"
        if session_id:
            query += " WHERE q.session_id = $session_id"
        query += """
            OPTIONAL MATCH (q)-[:USED_SECTION]->(s:Section)
            RETURN q, count(s) as sources_count
            ORDER BY q.created_at DESC
            LIMIT $limit
        """

        with self.driver.session(database=self.database) as session:
            result = session.run(query, session_id=session_id, limit=limit)
            return [{"question": dict(r["q"]), "sources_count": r["sources_count"]} for r in result]

    # ═══════════════════════════════════════════════════════════════════════════
    # 통계
    # ═══════════════════════════════════════════════════════════════════════════

    def get_graph_stats(self) -> Dict:
        """그래프 통계"""
        with self.driver.session(database=self.database) as session:
            result = session.run("""
                OPTIONAL MATCH (d:Document) WITH count(d) as docs
                OPTIONAL MATCH (s:Section) WITH docs, count(s) as sections
                OPTIONAL MATCH (dt:DocumentType) WITH docs, sections, count(dt) as doc_types
                OPTIONAL MATCH (c:Concept) WITH docs, sections, doc_types, count(c) as concepts
                OPTIONAL MATCH (q:Question) WITH docs, sections, doc_types, concepts, count(q) as questions
                OPTIONAL MATCH ()-[r]->() WITH docs, sections, doc_types, concepts, questions, count(r) as rels
                RETURN docs, sections, doc_types, concepts, questions, rels
            """)
            record = result.single()
            return {
                "documents": record["docs"] or 0,
                "sections": record["sections"] or 0,
                "document_types": record["doc_types"] or 0,
                "concepts": record["concepts"] or 0,
                "questions": record["questions"] or 0,
                "relationships": record["rels"] or 0
            }


# ═══════════════════════════════════════════════════════════════════════════
# 유틸리티: 문서 업로드 헬퍼
# ═══════════════════════════════════════════════════════════════════════════

def upload_document_to_graph(graph: Neo4jGraphStore, result: dict, filename: str):
    """document_pipeline 결과를 Neo4j에 업로드 (evaluate_gmp_unified 호환)"""
    doc_id = result.get("doc_id") or "UNKNOWN"
    title = result.get("doc_title") or filename

    # 문서 타입 추출 (EQ-SOP, EQ-WI 등)
    doc_type_code = ""
    doc_type_kr = ""
    doc_type_en = ""
    if doc_id.startswith("EQ-SOP"):
        doc_type_code = "SOP"
        doc_type_kr = "표준운영절차서"
        doc_type_en = "Standard Operating Procedure"
    elif doc_id.startswith("EQ-WI"):
        doc_type_code = "WI"
        doc_type_kr = "작업지침서"
        doc_type_en = "Work Instruction"
    elif doc_id.startswith("EQ-FORM"):
        doc_type_code = "FORM"
        doc_type_kr = "양식"
        doc_type_en = "Form"

    # Document 생성
    graph.create_document(doc_id=doc_id, title=title, version="1.0")

    # DocumentType 생성 및 연결
    if doc_type_code:
        graph.create_document_type(doc_type_code, doc_type_kr, doc_type_en)
        graph.link_document_type(doc_id, doc_type_code)

    # 기본 DocumentType 노드들 초기화 (MERGE이므로 중복 없음)
    doc_types = [
        ("SOP", "표준운영절차서", "Standard Operating Procedure"),
        ("WI", "작업지침서", "Work Instruction"),
        ("FORM", "양식", "Form"),
        ("MBR", "제조기록서", "Master Batch Record"),
        ("SPEC", "규격서", "Specification"),
    ]
    for code, name_kr, name_en in doc_types:
        graph.create_document_type(code, name_kr, name_en)

    # 기본 Concept 노드들 초기화 (MERGE이므로 중복 없음)
    concepts = [
        ("user_account", "사용자 접근 관리", "User Access Management", "사용자 계정, 권한, 역할 관리"),
        ("document_lifecycle", "문서 수명주기", "Document Lifecycle", "문서 작성, 승인, 개정, 폐기 등"),
        ("training", "교육 및 자격", "Training and Qualification", "교육, 훈련, 자격 관리"),
        ("system_configuration", "시스템 설정", "System Configuration", "시스템 구성 및 설정"),
        ("audit_evidence", "감사 증적", "Audit Evidence", "감사 대응 자료"),
    ]
    for concept_id, name_kr, name_en, description in concepts:
        graph.create_concept(concept_id, name_kr, name_en, description)

    # Section 생성
    for chunk in result.get("chunks", []):
        meta = chunk.get("metadata", {})
        clause_id = meta.get("clause_id")
        if not clause_id:
            continue

        # section_id는 문서ID:조항번호 형식으로 전역 고유하게
        section_id = f"{doc_id}:{clause_id}"
        main_section = clause_id.split('.')[0] if '.' in clause_id else clause_id

        # 모든 LLM 메타데이터 필드 포함
        llm_meta = {
            "content_type": meta.get("content_type", ""),
            "main_topic": meta.get("main_topic", ""),
            "sub_topics": meta.get("sub_topics", []),
            "actors": meta.get("actors", []),
            "actions": meta.get("actions", []),
            "conditions": meta.get("conditions", []),
            "summary": meta.get("summary", ""),
            "intent_scope": meta.get("intent_scope", ""),
            "intent_summary": meta.get("intent_summary", ""),
            "language": meta.get("language", "ko"),
        }

        graph.create_section(
            doc_id=doc_id,
            section_id=section_id,
            title=meta.get("title", ""),
            content=chunk.get("text", ""),
            clause_level=meta.get("clause_level", 0),
            main_section=main_section,
            llm_meta=llm_meta
        )

        # Concept 연결 (intent_scope가 있으면)
        intent_scope = llm_meta.get("intent_scope", "")
        if intent_scope:
            graph.link_section_concept(section_id, intent_scope)

        # 계층 관계 (부모도 문서ID 포함)
        if '.' in clause_id:
            parent_clause = '.'.join(clause_id.split('.')[:-1])
            parent_section_id = f"{doc_id}:{parent_clause}"
            graph.create_section_hierarchy(parent_section_id, section_id)

        # 타 문서/조항 언급 추출
        content = chunk.get("text", "")
        # 문서 ID 패턴 (EQ-SOP-00009, EQ-WI-00012 등)
        doc_mentions = re.findall(r'(EQ-[A-Z]+-\d{5})', content, re.IGNORECASE)
        if doc_mentions:
            graph.link_section_mentions(section_id, list(set(doc_mentions)))


# ═══════════════════════════════════════════════════════════════════════════
# Question 추적 헬퍼
# ═══════════════════════════════════════════════════════════════════════════

def track_rag_question(graph: Neo4jGraphStore, question_text: str,
                      search_results: List[Dict], answer: str = None,
                      session_id: str = None, llm_model: str = None) -> str:
    """RAG 질문 추적"""
    question_id = str(uuid.uuid4())

    # Question 생성
    graph.create_question(
        question_id=question_id,
        text=question_text,
        answer=answer,
        session_id=session_id,
        llm_model=llm_model
    )

    # 검색 결과 연결
    for rank, result in enumerate(search_results, start=1):
        meta = result.get("metadata", {})
        section_id = meta.get("section_id") or f"{meta.get('doc_id')}:{meta.get('clause_id')}"
        score = result.get("similarity", result.get("score", 0))

        if section_id:
            graph.link_question_to_section(question_id, section_id, rank, float(score))

    return question_id
