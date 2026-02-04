"""
Neo4j 그래프 저장소 모듈 v9.1

🔥 v9.1 추가:
- Question 노드: 사용자 질문 기록
- USED_SECTION 관계: 질문 답변 시 참조한 섹션 추적
- RAG 설명 가능성(Explainability) 강화

노드 타입:
- Document: 문서 고유 번호 (doc_id, title, version, effective_date, owning_dept)
- Section: 조항/절 (section_id, title, content, clause_level, main_section, embedding, content_type, main_topic, sub_topics, actors, actions, conditions, summary, intent_scope, intent_summary, intent_embedding, language)
- DocumentType: 문서 유형 (code, name_kr, name_en)
- Concept: 관리 개념 (concept_id, name_kr, name_en, description)
- Question: 사용자 질문 기록

관계 타입:
- HAS_SECTION: Document -> Section (문서가 포함하는 조항)
- PARENT_OF: Section -> Section (조항 간 상/하위 계층)
- REFERENCES: Document -> Document (문서 간 상호 참조)
- IS_TYPE: Document -> DocumentType (문서 유형 연결)
- MENTIONS: Section -> Document (조항 내 타 문서 언급)
- BELONGS_TO_CONCEPT: Section -> Concept (조항의 관리 영역 연결)
- USED_SECTION: Question -> Section (답변 시 참조한 섹션)
"""

from neo4j import GraphDatabase
from typing import List, Dict, Optional, Any
import re
import uuid
from datetime import datetime


class Neo4jGraphStore:
    """Neo4j 그래프 저장소 클래스"""
    
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
        is_aura = "databases.neo4j.io" in self.uri
        if is_aura:
            print(f"☁️ [Neo4j] Cloud(Aura) 연결 시도 중... ({self.uri})")
        else:
            print(f"🏠 [Neo4j] 로컬/원격 연결 시도 중... ({self.uri})")

        if not self.driver:
            try:
                self.driver = GraphDatabase.driver(
                    self.uri, 
                    auth=(self.user, self.password)
                )
                # 연결 즉시 확인
                if self.test_connection():
                    success_msg = "✅ Neo4j Aura 연결 성공!" if is_aura else f"✅ Neo4j 연결 성공! ({self.uri})"
                    print(success_msg)
            except Exception as e:
                print(f"❌ Neo4j 드라이버 생성 실패: {e}")
        return self
    
    def close(self):
        """연결 종료"""
        if self.driver:
            self.driver.close()
            self.driver = None
    
    def __enter__(self):
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    def test_connection(self) -> bool:
        """연결 테스트"""
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run("RETURN 'Connected!' AS message")
                record = result.single()
                print(f"✅ Neo4j 연결 성공: {record['message']}")
                return True
        except Exception as e:
            print(f"❌ Neo4j 연결 실패: {e}")
            return False
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 스키마 초기화
    # ═══════════════════════════════════════════════════════════════════════════
    
    def init_schema(self):
        """인덱스 및 제약조건 생성"""
        constraints = [
            "CREATE CONSTRAINT doc_id IF NOT EXISTS FOR (d:Document) REQUIRE d.doc_id IS UNIQUE",
            "CREATE CONSTRAINT section_id IF NOT EXISTS FOR (s:Section) REQUIRE s.section_id IS UNIQUE",
            "CREATE CONSTRAINT doctype_code IF NOT EXISTS FOR (dt:DocumentType) REQUIRE dt.code IS UNIQUE",
            "CREATE CONSTRAINT concept_id IF NOT EXISTS FOR (c:Concept) REQUIRE c.concept_id IS UNIQUE",
            "CREATE CONSTRAINT term_name IF NOT EXISTS FOR (t:Term) REQUIRE t.name IS UNIQUE",
            "CREATE CONSTRAINT question_id IF NOT EXISTS FOR (q:Question) REQUIRE q.id IS UNIQUE",
            "CREATE INDEX doc_title IF NOT EXISTS FOR (d:Document) ON (d.title)",
            "CREATE INDEX section_title IF NOT EXISTS FOR (s:Section) ON (s.title)",
            "CREATE INDEX section_main_topic IF NOT EXISTS FOR (s:Section) ON (s.main_topic)",
            "CREATE INDEX section_intent_scope IF NOT EXISTS FOR (s:Section) ON (s.intent_scope)",
            "CREATE INDEX question_created IF NOT EXISTS FOR (q:Question) ON (q.created_at)",
        ]
        
        with self.driver.session(database=self.database) as session:
            for constraint in constraints:
                try:
                    session.run(constraint)
                except Exception as e:
                    print(f"⚠️ 제약조건 생성 스킵: {e}")
        
        print("✅ 스키마 초기화 완료")
    
    def clear_all(self):
        """모든 노드와 관계 삭제"""
        with self.driver.session(database=self.database) as session:
            session.run("MATCH (n) DETACH DELETE n")
        print("🗑️ 모든 데이터 삭제 완료")
    
    def delete_document(self, sop_id: str) -> Dict:
        """
        특정 문서와 관련된 모든 노드/관계 삭제
        
        삭제 대상:
        - Document 노드
        - 연결된 Section 노드들
        - 연결된 Term 노드들 (다른 문서에서 사용 안 하면)
        - 연결된 Role 노드들 (다른 문서에서 사용 안 하면)
        - 관련 Question의 USED_SECTION 관계 (Question은 유지)
        """
        deleted = {"document": 0, "sections": 0, "terms": 0, "roles": 0, "relations": 0}
        
        with self.driver.session(database=self.database) as session:
            # 1. Section 삭제 (USED_SECTION 관계도 함께 삭제됨)
            result = session.run("""
                MATCH (s:Section {doc_sop_id: $sop_id})
                DETACH DELETE s
                RETURN count(s) as count
            """, sop_id=sop_id)
            record = result.single()
            deleted["sections"] = record["count"] if record else 0
            
            # 2. 고아 Term 삭제 (다른 문서에서 참조 안 하는 것만)
            result = session.run("""
                MATCH (t:Term)
                WHERE NOT EXISTS { MATCH (:Document)-[:DEFINES]->(t) }
                DETACH DELETE t
                RETURN count(t) as count
            """)
            record = result.single()
            deleted["terms"] = record["count"] if record else 0
            
            # 3. 고아 Role 삭제
            result = session.run("""
                MATCH (r:Role)
                WHERE NOT EXISTS { MATCH (:Document)-[:ASSIGNS]->(r) }
                DETACH DELETE r
                RETURN count(r) as count
            """)
            record = result.single()
            deleted["roles"] = record["count"] if record else 0
            
            # 4. Document 삭제
            result = session.run("""
                MATCH (d:Document {sop_id: $sop_id})
                DETACH DELETE d
                RETURN count(d) as count
            """, sop_id=sop_id)
            record = result.single()
            deleted["document"] = record["count"] if record else 0
        
        print(f"🗑️ Neo4j 문서 삭제: {sop_id} - {deleted}")
        return deleted
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 문서 노드 생성
    # ═══════════════════════════════════════════════════════════════════════════
    
    def create_document(
        self,
        doc_id: str,
        title: str,
        version: str = "1.0",
        effective_date: str = None,
        owning_dept: str = None,
        metadata: Dict = None
    ) -> Dict:
        """Document 노드 생성"""
        query = """
        MERGE (d:Document {doc_id: $doc_id})
        SET d.title = $title,
            d.version = $version,
            d.effective_date = $effective_date,
            d.owning_dept = $owning_dept,
            d.metadata = $metadata,
            d.updated_at = datetime()
        RETURN d
        """
        
        with self.driver.session(database=self.database) as session:
            result = session.run(
                query,
                doc_id=doc_id,
                title=title,
                version=version,
                effective_date=effective_date,
                owning_dept=owning_dept,
                metadata=str(metadata or {})
            )
            record = result.single()
            return dict(record["d"]) if record else None
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 섹션 노드 생성
    # ═══════════════════════════════════════════════════════════════════════════
    
    def create_section(
        self,
        doc_id: str,
        section_id: str,
        title: str,
        content: str,
        clause_level: int = 0,
        main_section: str = None,
        embedding: List[float] = None,
        llm_meta: Dict = None,
        page: int = None
    ) -> Dict:
        """Section 노드 생성 및 Document와 연결"""
        query = """
        MATCH (d:Document {doc_id: $doc_id})
        MERGE (s:Section {section_id: $section_id})
        SET s.title = $title,
            s.content = $content,
            s.clause_level = $clause_level,
            s.main_section = $main_section,
            s.embedding = $embedding,
            s.content_type = $content_type,
            s.main_topic = $main_topic,
            s.sub_topics = $sub_topics,
            s.actors = $actors,
            s.actions = $actions,
            s.conditions = $conditions,
            s.summary = $summary,
            s.intent_scope = $intent_scope,
            s.intent_summary = $intent_summary,
            s.intent_embedding = $intent_embedding,
            s.language = $language,
            s.page = $page,
            s.doc_id = $doc_id
        MERGE (d)-[:HAS_SECTION]->(s)
        RETURN s
        """
        
        meta = llm_meta or {}
        with self.driver.session(database=self.database) as session:
            result = session.run(
                query,
                doc_id=doc_id,
                section_id=section_id,
                title=title,
                content=content,
                clause_level=clause_level,
                main_section=main_section,
                embedding=embedding,
                content_type=meta.get("content_type"),
                main_topic=meta.get("main_topic"),
                sub_topics=meta.get("sub_topics"), # 쉼표 구분 문자열 기대
                actors=meta.get("actors"),
                actions=meta.get("actions"),
                conditions=meta.get("conditions"),
                summary=meta.get("summary"),
                intent_scope=meta.get("intent_scope"),
                intent_summary=meta.get("intent_summary"),
                intent_embedding=meta.get("intent_embedding"),
                language=meta.get("language", "ko"),
                page=page
            )
            record = result.single()
            return dict(record["s"]) if record else None
    
    def create_section_hierarchy(
        self,
        parent_section_id: str,
        child_section_id: str
    ):
        """섹션 간 계층 관계 생성"""
        query = """
        MATCH (parent:Section {section_id: $parent_id})
        MATCH (child:Section {section_id: $child_id})
        MERGE (parent)-[:PARENT_OF]->(child)
        """
        
        with self.driver.session(database=self.database) as session:
            session.run(
                query,
                parent_id=parent_section_id,
                child_id=child_section_id
            )
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 🆕 Question 노드 (질문 추적)
    # ═══════════════════════════════════════════════════════════════════════════
    
    def create_question(
        self,
        question_id: str,
        text: str,
        answer: str = None,
        session_id: str = None,
        embedding_model: str = None,
        llm_model: str = None
    ) -> Dict:
        """
        Question 노드 생성
        
        Args:
            question_id: 고유 ID (UUID)
            text: 질문 텍스트
            answer: LLM 답변 (나중에 업데이트 가능)
            session_id: 대화 세션 ID
            embedding_model: 사용된 임베딩 모델
            llm_model: 사용된 LLM 모델
        """
        query = """
        MERGE (q:Question {id: $id})
        SET q.text = $text,
            q.answer = $answer,
            q.session_id = $session_id,
            q.embedding_model = $embedding_model,
            q.llm_model = $llm_model,
            q.created_at = datetime()
        RETURN q
        """
        
        with self.driver.session(database=self.database) as session:
            result = session.run(
                query,
                id=question_id,
                text=text,
                answer=answer,
                session_id=session_id,
                embedding_model=embedding_model,
                llm_model=llm_model
            )
            record = result.single()
            return dict(record["q"]) if record else None
    
    def update_question_answer(
        self,
        question_id: str,
        answer: str
    ):
        """질문에 답변 업데이트"""
        query = """
        MATCH (q:Question {id: $id})
        SET q.answer = $answer,
            q.answered_at = datetime()
        RETURN q
        """
        
        with self.driver.session(database=self.database) as session:
            result = session.run(query, id=question_id, answer=answer)
            record = result.single()
            return dict(record["q"]) if record else None
    
    def link_question_to_section(
        self,
        question_id: str,
        sop_id: str,
        section_id: str,
        rank: int = None,
        score: float = None
    ):
        """
        Question과 참조된 Section 연결 (USED_SECTION 관계)
        
        Args:
            question_id: 질문 ID
            sop_id: 문서 SOP ID
            section_id: 섹션 ID
            rank: 검색 순위 (1 = 가장 관련성 높음)
            score: 유사도 점수 (0~1)
        """
        query = """
        MATCH (q:Question {id: $question_id})
        MATCH (s:Section {doc_sop_id: $sop_id, section_id: $section_id})
        MERGE (q)-[r:USED_SECTION]->(s)
        SET r.rank = $rank,
            r.score = $score,
            r.used_at = datetime()
        RETURN r
        """
        
        with self.driver.session(database=self.database) as session:
            session.run(
                query,
                question_id=question_id,
                sop_id=sop_id,
                section_id=section_id,
                rank=rank,
                score=score
            )
    
    def link_question_to_document(
        self,
        question_id: str,
        doc_id: str
    ):
        """Question과 관련 Document 연결 (ABOUT_DOCUMENT 관계)"""
        query = """
        MATCH (q:Question {id: $question_id})
        MATCH (d:Document {doc_id: $doc_id})
        MERGE (q)-[r:ABOUT_DOCUMENT]->(d)
        SET r.linked_at = datetime()
        RETURN r
        """
        
        with self.driver.session(database=self.database) as session:
            session.run(
                query,
                question_id=question_id,
                doc_id=doc_id
            )
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 신규 노드 및 관계 기능
    # ═══════════════════════════════════════════════════════════════════════════
    
    def create_document_type(self, code: str, name_kr: str, name_en: str):
        """DocumentType 노드 생성"""
        query = """
        MERGE (dt:DocumentType {code: $code})
        SET dt.name_kr = $name_kr,
            dt.name_en = $name_en
        RETURN dt
        """
        with self.driver.session(database=self.database) as session:
            session.run(query, code=code, name_kr=name_kr, name_en=name_en)

    def link_doc_to_type(self, doc_id: str, type_code: str):
        """Document -> DocumentType 연결 (IS_TYPE)"""
        query = """
        MATCH (d:Document {doc_id: $doc_id})
        MATCH (dt:DocumentType {code: $type_code})
        MERGE (d)-[:IS_TYPE]->(dt)
        """
        with self.driver.session(database=self.database) as session:
            session.run(query, doc_id=doc_id, type_code=type_code)

    def create_concept(self, concept_id: str, name_kr: str, name_en: str, description: str = None):
        """Concept 노드 생성"""
        query = """
        MERGE (c:Concept {concept_id: $concept_id})
        SET c.name_kr = $name_kr,
            c.name_en = $name_en,
            c.description = $description
        RETURN c
        """
        with self.driver.session(database=self.database) as session:
            session.run(query, concept_id=concept_id, name_kr=name_kr, name_en=name_en, description=description)

    def link_section_to_concept(self, section_id: str, concept_id: str):
        """Section -> Concept 연결 (BELONGS_TO_CONCEPT)"""
        query = """
        MATCH (s:Section {section_id: $section_id})
        MATCH (c:Concept {concept_id: $concept_id})
        MERGE (s)-[:BELONGS_TO_CONCEPT]->(c)
        """
        with self.driver.session(database=self.database) as session:
            session.run(query, section_id=section_id, concept_id=concept_id)

    def link_section_to_mention_doc(self, section_id: str, mentioned_doc_id: str):
        """Section -> Document 연결 (MENTIONS)"""
        query = """
        MATCH (s:Section {section_id: $section_id})
        MERGE (d:Document {doc_id: $mentioned_doc_id})
        MERGE (s)-[:MENTIONS]->(d)
        """
        with self.driver.session(database=self.database) as session:
            session.run(query, section_id=section_id, mentioned_doc_id=mentioned_doc_id)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 🆕 Question 조회 (RAG 설명 가능성)
    # ═══════════════════════════════════════════════════════════════════════════
    
    def get_question_sources(self, question_id: str) -> Dict:
        """
        질문이 어떤 Section을 참조해서 답변했는지 조회
        
        Returns:
            {
                "question": {...},
                "sources": [
                    {"section": {...}, "rank": 1, "score": 0.87},
                    ...
                ]
            }
        """
        query = """
        MATCH (q:Question {id: $id})
        OPTIONAL MATCH (q)-[r:USED_SECTION]->(s:Section)
        OPTIONAL MATCH (s)<-[:HAS_SECTION]-(d:Document)
        RETURN q,
               collect({
                   section: s,
                   document: d,
                   rank: r.rank,
                   score: r.score
               }) as sources
        ORDER BY r.rank
        """
        
        with self.driver.session(database=self.database) as session:
            result = session.run(query, id=question_id)
            record = result.single()
            if record:
                return {
                    "question": dict(record["q"]),
                    "sources": [
                        {
                            "section": dict(s["section"]) if s["section"] else None,
                            "document": dict(s["document"]) if s["document"] else None,
                            "rank": s["rank"],
                            "score": s["score"]
                        }
                        for s in record["sources"]
                        if s["section"]
                    ]
                }
            return None
    
    def get_section_usage_stats(self, sop_id: str = None) -> List[Dict]:
        """
        각 Section이 몇 번 참조되었는지 통계
        
        Returns:
            [
                {"section_id": "4.1", "sop_id": "EQ-SOP-00010", "usage_count": 15},
                ...
            ]
        """
        if sop_id:
            query = """
            MATCH (s:Section {doc_sop_id: $sop_id})<-[r:USED_SECTION]-(q:Question)
            RETURN s.section_id as section_id, 
                   s.doc_sop_id as sop_id,
                   s.name as section_name,
                   count(r) as usage_count,
                   avg(r.score) as avg_score
            ORDER BY usage_count DESC
            """
            params = {"sop_id": sop_id}
        else:
            query = """
            MATCH (s:Section)<-[r:USED_SECTION]-(q:Question)
            RETURN s.section_id as section_id,
                   s.doc_sop_id as sop_id,
                   s.name as section_name,
                   count(r) as usage_count,
                   avg(r.score) as avg_score
            ORDER BY usage_count DESC
            LIMIT 50
            """
            params = {}
        
        with self.driver.session(database=self.database) as session:
            result = session.run(query, **params)
            return [dict(record) for record in result]
    
    def get_questions_by_section(
        self,
        sop_id: str,
        section_id: str,
        limit: int = 20
    ) -> List[Dict]:
        """특정 Section을 참조한 질문 목록"""
        query = """
        MATCH (q:Question)-[r:USED_SECTION]->(s:Section {doc_sop_id: $sop_id, section_id: $section_id})
        RETURN q.id as question_id,
               q.text as question_text,
               q.answer as answer,
               r.rank as rank,
               r.score as score,
               q.created_at as created_at
        ORDER BY q.created_at DESC
        LIMIT $limit
        """
        
        with self.driver.session(database=self.database) as session:
            result = session.run(
                query,
                sop_id=sop_id,
                section_id=section_id,
                limit=limit
            )
            return [dict(record) for record in result]
    
    def get_similar_questions(
        self,
        text_contains: str,
        limit: int = 10
    ) -> List[Dict]:
        """특정 키워드를 포함하는 질문 검색"""
        query = """
        MATCH (q:Question)
        WHERE toLower(q.text) CONTAINS toLower($keyword)
        OPTIONAL MATCH (q)-[r:USED_SECTION]->(s:Section)
        RETURN q.id as question_id,
               q.text as question_text,
               q.answer as answer,
               collect(s.section_id) as used_sections
        ORDER BY q.created_at DESC
        LIMIT $limit
        """
        
        with self.driver.session(database=self.database) as session:
            result = session.run(query, keyword=text_contains, limit=limit)
            return [dict(record) for record in result]
    
    def get_question_history(
        self,
        session_id: str = None,
        limit: int = 50
    ) -> List[Dict]:
        """질문 히스토리 조회"""
        if session_id:
            query = """
            MATCH (q:Question {session_id: $session_id})
            OPTIONAL MATCH (q)-[:USED_SECTION]->(s:Section)
            RETURN q.id as question_id,
                   q.text as question_text,
                   q.answer as answer,
                   q.created_at as created_at,
                   count(s) as sources_count
            ORDER BY q.created_at DESC
            LIMIT $limit
            """
            params = {"session_id": session_id, "limit": limit}
        else:
            query = """
            MATCH (q:Question)
            OPTIONAL MATCH (q)-[:USED_SECTION]->(s:Section)
            RETURN q.id as question_id,
                   q.text as question_text,
                   q.answer as answer,
                   q.created_at as created_at,
                   q.session_id as session_id,
                   count(s) as sources_count
            ORDER BY q.created_at DESC
            LIMIT $limit
            """
            params = {"limit": limit}
        
        with self.driver.session(database=self.database) as session:
            result = session.run(query, **params)
            return [dict(record) for record in result]
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 용어 노드 생성
    # ═══════════════════════════════════════════════════════════════════════════
    
    def create_term(
        self,
        name: str,
        definition: str,
        english_name: str = None,
        sop_id: str = None
    ) -> Dict:
        """Term 노드 생성"""
        query = """
        MERGE (t:Term {name: $name})
        SET t.definition = $definition,
            t.english_name = $english_name
        WITH t
        OPTIONAL MATCH (d:Document {sop_id: $sop_id})
        FOREACH (_ IN CASE WHEN d IS NOT NULL THEN [1] ELSE [] END |
            MERGE (d)-[:DEFINES]->(t)
        )
        RETURN t
        """
        
        with self.driver.session(database=self.database) as session:
            result = session.run(
                query,
                name=name,
                definition=definition[:1000] if definition else "",
                english_name=english_name,
                sop_id=sop_id
            )
            record = result.single()
            return dict(record["t"]) if record else None
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 역할 노드 생성
    # ═══════════════════════════════════════════════════════════════════════════
    
    def create_role(
        self,
        name: str,
        responsibilities: str,
        sop_id: str = None
    ) -> Dict:
        """Role 노드 생성"""
        query = """
        MERGE (r:Role {name: $name})
        SET r.responsibilities = $responsibilities
        WITH r
        OPTIONAL MATCH (d:Document {sop_id: $sop_id})
        FOREACH (_ IN CASE WHEN d IS NOT NULL THEN [1] ELSE [] END |
            MERGE (d)-[:ASSIGNS]->(r)
        )
        RETURN r
        """
        
        with self.driver.session(database=self.database) as session:
            result = session.run(
                query,
                name=name,
                responsibilities=responsibilities[:2000] if responsibilities else "",
                sop_id=sop_id
            )
            record = result.single()
            return dict(record["r"]) if record else None
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 문서 간 참조 관계
    # ═══════════════════════════════════════════════════════════════════════════
    
    def create_reference(
        self,
        from_sop_id: str,
        to_sop_id: str,
        reference_type: str = "참조"
    ):
        """문서 간 참조 관계 생성"""
        query = """
        MERGE (from:Document {sop_id: $from_id})
        MERGE (to:Document {sop_id: $to_id})
        MERGE (from)-[r:REFERENCES]->(to)
        SET r.type = $ref_type
        """
        
        with self.driver.session(database=self.database) as session:
            session.run(
                query,
                from_id=from_sop_id,
                to_id=to_sop_id,
                ref_type=reference_type
            )
    
    def link_section_to_doc(
        self,
        sop_id: str,
        section_id: str,
        ref_sop_id: str
    ):
        """섹션에서 문서로의 참조 관계 생성"""
        query = """
        MATCH (s:Section {doc_sop_id: $sop_id, section_id: $section_id})
        MERGE (target:Document {sop_id: $ref_id})
        MERGE (s)-[:REFERENCES_DOC]->(target)
        """
        
        with self.driver.session(database=self.database) as session:
            session.run(
                query,
                sop_id=sop_id,
                section_id=section_id,
                ref_id=ref_sop_id
            )
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 검색 기능
    # ═══════════════════════════════════════════════════════════════════════════
    
    def search_sections(
        self,
        keyword: str,
        sop_id: str = None,
        limit: int = 10
    ) -> List[Dict]:
        """섹션 내용 검색"""
        if sop_id:
            query = """
            MATCH (s:Section {doc_sop_id: $sop_id})
            WHERE toLower(s.content) CONTAINS toLower($keyword)
               OR toLower(s.name) CONTAINS toLower($keyword)
            RETURN s
            LIMIT $limit
            """
            params = {"sop_id": sop_id, "keyword": keyword, "limit": limit}
        else:
            query = """
            MATCH (s:Section)
            WHERE toLower(s.content) CONTAINS toLower($keyword)
               OR toLower(s.name) CONTAINS toLower($keyword)
            RETURN s
            LIMIT $limit
            """
            params = {"keyword": keyword, "limit": limit}
        
        with self.driver.session(database=self.database) as session:
            result = session.run(query, **params)
            return [dict(record["s"]) for record in result]
    
    def search_by_term(self, term: str) -> List[Dict]:
        """용어로 관련 문서/섹션 검색"""
        query = """
        MATCH (t:Term)
        WHERE toLower(t.name) CONTAINS toLower($term)
           OR toLower(t.english_name) CONTAINS toLower($term)
        OPTIONAL MATCH (d:Document)-[:DEFINES]->(t)
        RETURN t, collect(DISTINCT d.sop_id) as documents
        """
        
        with self.driver.session(database=self.database) as session:
            result = session.run(query, term=term)
            return [
                {
                    "term": dict(record["t"]),
                    "documents": list(record["documents"])
                }
                for record in result
            ]
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 조회 기능
    # ═══════════════════════════════════════════════════════════════════════════
    
    def get_all_documents(self) -> List[Dict]:
        """모든 문서 목록"""
        query = """
        MATCH (d:Document)
        OPTIONAL MATCH (d)-[:HAS_SECTION]->(s:Section)
        RETURN d, count(s) as section_count
        ORDER BY d.sop_id
        """
        
        with self.driver.session(database=self.database) as session:
            result = session.run(query)
            return [
                {
                    **dict(record["d"]),
                    "section_count": record["section_count"]
                }
                for record in result
            ]
    
    def get_document(self, sop_id: str) -> Optional[Dict]:
        """특정 문서 상세 조회"""
        query = """
        MATCH (d:Document {sop_id: $sop_id})
        OPTIONAL MATCH (d)-[:HAS_SECTION]->(s:Section)
        OPTIONAL MATCH (d)-[:DEFINES]->(t:Term)
        OPTIONAL MATCH (d)-[:ASSIGNS]->(r:Role)
        RETURN d,
               collect(DISTINCT {id: s.section_id, name: s.name, type: s.section_type}) as sections,
               collect(DISTINCT t.name) as terms,
               collect(DISTINCT r.name) as roles
        """
        
        with self.driver.session(database=self.database) as session:
            result = session.run(query, sop_id=sop_id)
            record = result.single()
            if record:
                return {
                    "document": dict(record["d"]),
                    "sections": list(record["sections"]),
                    "terms": list(record["terms"]),
                    "roles": list(record["roles"])
                }
            return None
    
    def get_document_references(self, sop_id: str) -> Optional[Dict]:
        """문서 참조 관계"""
        query = """
        MATCH (d:Document {sop_id: $sop_id})
        OPTIONAL MATCH (d)-[:REFERENCES]->(ref:Document)
        OPTIONAL MATCH (citing:Document)-[:REFERENCES]->(d)
        RETURN d,
               collect(DISTINCT ref.sop_id) as references,
               collect(DISTINCT citing.sop_id) as cited_by
        """
        
        with self.driver.session(database=self.database) as session:
            result = session.run(query, sop_id=sop_id)
            record = result.single()
            if record:
                return {
                    "document": dict(record["d"]),
                    "references": list(record["references"]),
                    "cited_by": list(record["cited_by"])
                }
            return None
    
    def get_section_hierarchy(self, sop_id: str) -> List[Dict]:
        """문서의 섹션 계층 구조"""
        query = """
        MATCH (d:Document {sop_id: $sop_id})-[:HAS_SECTION]->(s:Section)
        OPTIONAL MATCH (s)-[:PARENT_OF]->(child:Section)
        RETURN s, collect(child.section_id) as children
        ORDER BY s.section_path
        """
        
        with self.driver.session(database=self.database) as session:
            result = session.run(query, sop_id=sop_id)
            return [
                {
                    "section": dict(record["s"]),
                    "children": list(record["children"])
                }
                for record in result
            ]
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 그래프 분석
    # ═══════════════════════════════════════════════════════════════════════════
    
    def get_graph_stats(self) -> Dict:
        """그래프 통계 (없는 라벨도 처리)"""
        query = """
        OPTIONAL MATCH (d:Document) WITH count(d) as docs
        OPTIONAL MATCH (s:Section) WITH docs, count(s) as sections
        OPTIONAL MATCH (t:Term) WITH docs, sections, count(t) as terms
        OPTIONAL MATCH (r:Role) WITH docs, sections, terms, count(r) as roles
        OPTIONAL MATCH (q:Question) WITH docs, sections, terms, roles, count(q) as questions
        OPTIONAL MATCH ()-[rel]->() WITH docs, sections, terms, roles, questions, count(rel) as rels
        RETURN docs, sections, terms, roles, questions, rels
        """
        
        with self.driver.session(database=self.database) as session:
            result = session.run(query)
            record = result.single()
            if record:
                return {
                    "documents": record["docs"] or 0,
                    "sections": record["sections"] or 0,
                    "terms": record["terms"] or 0,
                    "roles": record["roles"] or 0,
                    "questions": record["questions"] or 0,
                    "relationships": record["rels"] or 0
                }
            return {}


# ═══════════════════════════════════════════════════════════════════════════
# 🆕 RAG 질문 추적 헬퍼 함수
# ═══════════════════════════════════════════════════════════════════════════

def track_rag_question(
    graph_store: Neo4jGraphStore,
    question_text: str,
    search_results: List[Dict],
    answer: str = None,
    session_id: str = None,
    embedding_model: str = None,
    llm_model: str = None
) -> str:
    """
    RAG 질문과 참조된 섹션을 Neo4j에 기록
    
    Args:
        graph_store: Neo4jGraphStore 인스턴스
        question_text: 사용자 질문
        search_results: 벡터 검색 결과 [{metadata: {sop_id, section, ...}, similarity: 0.87}, ...]
        answer: LLM 답변
        session_id: 대화 세션 ID
        embedding_model: 사용된 임베딩 모델
        llm_model: 사용된 LLM 모델
    
    Returns:
        question_id: 생성된 질문 ID
    """
    question_id = str(uuid.uuid4())
    
    # 1. Question 노드 생성
    graph_store.create_question(
        question_id=question_id,
        text=question_text,
        answer=answer,
        session_id=session_id,
        embedding_model=embedding_model,
        llm_model=llm_model
    )
    
    # 2. 검색 결과에서 섹션 연결
    seen_sections = set()  # 중복 방지
    
    for rank, result in enumerate(search_results, start=1):
        meta = result.get("metadata", {})
        sop_id = meta.get("sop_id")
        
        # section_id 결정 (article_num 또는 section에서 추출)
        section_id = meta.get("article_num")
        if not section_id:
            section = meta.get("section", "")
            # "5.1 품질관리기준서..." → "5.1" 추출
            match = re.match(r'^(\d+(?:\.\d+)*)', section)
            if match:
                section_id = match.group(1)
        
        if not sop_id or not section_id:
            continue
        
        # 중복 방지
        key = f"{sop_id}:{section_id}"
        if key in seen_sections:
            continue
        seen_sections.add(key)
        
        # USED_SECTION 관계 생성
        score = result.get("similarity", result.get("score"))
        graph_store.link_question_to_section(
            question_id=question_id,
            sop_id=sop_id,
            section_id=section_id,
            rank=rank,
            score=float(score) if score else None
        )
        
        # ABOUT_DOCUMENT 관계 생성 (첫 번째 문서만)
        if rank == 1:
            graph_store.link_question_to_document(
                question_id=question_id,
                sop_id=sop_id
            )
    
    return question_id


def rebuild_document_references(graph_store: Neo4jGraphStore):
    """
    이미 등록된 모든 문서들 사이의 참조 관계를 재구축
    
    - 각 문서의 모든 섹션 텍스트를 스캔
    - 참조된 SOP ID를 찾아 관계 생성
    """
    print("🔄 문서 간 참조 관계 재구축 시작...")
    
    # 1. 모든 문서 조회
    documents = graph_store.get_all_documents()
    print(f"   📄 총 {len(documents)}개 문서 발견")
    
    total_refs = 0
    
    for doc in documents:
        sop_id = doc["sop_id"]
        print(f"   처리 중: {sop_id}")
        
        # 2. 해당 문서의 모든 섹션 조회
        sections = graph_store.search_sections("", sop_id=sop_id, limit=1000)  # 모든 섹션
        
        doc_refs = set()
        
        for section in sections:
            content = section.get("content", "")
            refs = extract_references_from_text(content)
            for ref in refs:
                if ref != sop_id:  # 자기 자신 제외
                    doc_refs.add(ref)
        
        # 3. 참조 관계 생성
        for ref_sop_id in doc_refs:
            graph_store.create_reference(sop_id, ref_sop_id)
            total_refs += 1
        
        if doc_refs:
            print(f"     → 참조: {list(doc_refs)}")
    
    print(f"✅ 참조 관계 재구축 완료: {total_refs}개 관계 생성")


# ═══════════════════════════════════════════════════════════════════════════
# 문서 파싱 → 그래프 변환
# ═══════════════════════════════════════════════════════════════════════════

def extract_terms_from_text(text: str) -> List[Dict]:
    """정의 섹션에서 용어 추출"""
    terms = []
    
    patterns = [
        r'^([가-힣A-Za-z\s]+)\s*\(([A-Za-z\s]+)\)\s*[:：]\s*(.+)',
        r'^([가-힣]+)\s*[:：]\s*(.+)',
    ]
    
    lines = text.split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        for i, pattern in enumerate(patterns):
            match = re.match(pattern, line)
            if match:
                if i == 0:
                    terms.append({
                        "name": match.group(1).strip(),
                        "english_name": match.group(2).strip(),
                        "definition": match.group(3).strip()
                    })
                else:
                    terms.append({
                        "name": match.group(1).strip(),
                        "english_name": None,
                        "definition": match.group(2).strip()
                    })
                break
    
    return terms


def extract_references_from_text(text: str) -> List[str]:
    """텍스트에서 SOP 참조 추출"""
    pattern = r'[A-Z]{2,}-?SOP-?\d{4,5}'
    matches = re.findall(pattern, text, re.IGNORECASE)
    
    normalized = set()
    for m in matches:
        m = m.upper().replace('_', '-')
        if not m.startswith('EQ-'):
            m = 'EQ-' + m
        normalized.add(m)
    
    return list(normalized)


def document_to_graph(
    graph_store: Neo4jGraphStore,
    parsed_doc,
    sop_id: str = None
):
    """ParsedDocument를 Neo4j 그래프로 변환"""
    
    sop_id = sop_id or parsed_doc.metadata.get("sop_id") or "UNKNOWN"
    title = parsed_doc.metadata.get("title") or parsed_doc.metadata.get("file_name") or "문서"
    version = parsed_doc.metadata.get("version") or "1.0"
    
    print(f"\n📄 문서 그래프 생성: {sop_id} - {title}")
    
    # 1. Document 노드 생성
    graph_store.create_document(
        sop_id=sop_id,
        title=title,
        version=version,
        doc_type="SOP",
        metadata=parsed_doc.metadata
    )
    
    # 2. 블록 → Section 노드 변환
    section_stack = {}
    
    for block in parsed_doc.blocks:
        meta = block.metadata
        section_id = meta.get("article_num") or meta.get("title") or "intro"
        section_type = meta.get("article_type", "intro")
        section_name = meta.get("title", "")
        section_path = meta.get("section_path")
        section_path_readable = meta.get("section_path_readable")
        page = getattr(block, 'page', None)
        
        # Section 노드 생성
        graph_store.create_section(
            sop_id=sop_id,
            section_id=str(section_id),
            name=section_name,
            section_type=section_type,
            content=block.text,
            section_path=section_path,
            section_path_readable=section_path_readable,
            page=page
        )
        
        # 계층 관계 설정
        if section_type == "subsection" and section_stack.get("section"):
            graph_store.create_section_hierarchy(
                sop_id=sop_id,
                parent_section_id=section_stack["section"],
                child_section_id=str(section_id)
            )
        elif section_type == "subsubsection" and section_stack.get("subsection"):
            graph_store.create_section_hierarchy(
                sop_id=sop_id,
                parent_section_id=section_stack["subsection"],
                child_section_id=str(section_id)
            )
        elif section_type == "level" and section_stack.get("named_section"):
            graph_store.create_section_hierarchy(
                sop_id=sop_id,
                parent_section_id=section_stack["named_section"],
                child_section_id=str(section_id)
            )
        
        # 스택 업데이트
        if section_type in ["section", "named_section"]:
            section_stack["section"] = str(section_id)
            section_stack["named_section"] = str(section_id)
            section_stack["subsection"] = None
            section_stack["subsubsection"] = None
        elif section_type in ["subsection", "level"]:
            section_stack["subsection"] = str(section_id)
            section_stack["subsubsection"] = None
        elif section_type == "subsubsection":
            section_stack["subsubsection"] = str(section_id)
        
        # 3. 🆕 섹션 텍스트에서 참조 추출 (섹션 단위 연결)
        section_refs = extract_references_from_text(block.text)
        for ref_id in section_refs:
            if ref_id != sop_id:
                graph_store.link_section_to_doc(sop_id, str(section_id), ref_id)
        
        # 4. 정의 섹션에서 용어 추출
        if section_name and ("정의" in section_name or "Definition" in section_name):
            terms = extract_terms_from_text(block.text)
            for term in terms:
                graph_store.create_term(
                    name=term["name"],
                    definition=term["definition"],
                    english_name=term.get("english_name"),
                    sop_id=sop_id
                )
            print(f"   📖 용어 {len(terms)}개 추출")
    
    # 5. 문서 전체 참조 추출 (섹션 단위 연결 보조)
    all_refs = extract_references_from_text(parsed_doc.text)
    for ref_sop_id in all_refs:
        if ref_sop_id != sop_id:
            graph_store.create_reference(sop_id, ref_sop_id)
    
    if all_refs:
        print(f"   🔗 참조 문서: {all_refs}")
    
    print(f"   ✅ 섹션 {len(parsed_doc.blocks)}개 생성 완료")


# ═══════════════════════════════════════════════════════════════════════════
# CLI 테스트
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    with Neo4jGraphStore() as graph:
        graph.test_connection()
        graph.init_schema()
        
        stats = graph.get_graph_stats()
        print(f"\n📊 그래프 통계: {stats}")
        
        # 🔥 문서 간 참조 관계 재구축
        print("\n🔗 문서 간 참조 관계 재구축...")
        rebuild_document_references(graph)
        
        # Question 추적 테스트
        print("\n🔥 Question 추적 테스트...")
        
        # 테스트 질문 생성
        test_q_id = track_rag_question(
            graph_store=graph,
            question_text="품질관리책임자의 역할은 무엇인가요?",
            search_results=[
                {
                    "metadata": {"sop_id": "EQ-SOP-00010", "article_num": "4.1", "section": "4.1 품질관리책임자"},
                    "similarity": 0.87
                },
                {
                    "metadata": {"sop_id": "EQ-SOP-00010", "article_num": "4.2", "section": "4.2 품질보증부서"},
                    "similarity": 0.72
                }
            ],
            answer="품질관리책임자(QC Manager)는 본 기준서의 제·개정을 담당합니다.",
            session_id="test-session-001",
            llm_model="qwen2.5:3b"
        )
        
        print(f"   생성된 Question ID: {test_q_id}")
        
        # 조회 테스트
        sources = graph.get_question_sources(test_q_id)
        print(f"   참조된 섹션: {len(sources['sources'])}개")
        
        # 통계 갱신
        stats = graph.get_graph_stats()
        print(f"\n📊 최종 통계: {stats}")