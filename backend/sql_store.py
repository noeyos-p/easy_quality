import psycopg2
from psycopg2.extras import RealDictCursor
import json
import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# DB 접속 정보 (환경변수 또는 요쳥된 기본값)
DB_CONFIG = {
    "host": os.getenv("PG_HOST", "localhost"),
    "database": os.getenv("PG_DATABASE", "postgres"),
    "user": os.getenv("PG_USER", "postgres"),
    "password": os.getenv("PG_PASSWORD", "1111"),
    "port": os.getenv("PG_PORT", "5432")
}

class SQLStore:
    """PostgreSQL 기반 원본 문서 및 메타데이터 저장소"""
    
    def __init__(self, config: Dict = None):
        self.config = config or DB_CONFIG
        
    def _get_connection(self):
        return psycopg2.connect(**self.config)

    def init_db(self):
        """스키마 초기화: 문서 기반 통합 관리 테이블 생성"""
        # 연결 정보 출력
        host = self.config.get("host", "localhost")
        if host in ["localhost", "127.0.0.1"]:
            print(f"🏠 [SQLStore] PostgreSQL: 로컬호스트 연결 중 ({host})")
        else:
            print(f"🌐 [SQLStore] PostgreSQL: 원격 DB 연결 중 ({host})")
            
        # sop_id의 UNIQUE 제약조건을 제거하고 (sop_id, version) 복합 유니크를 권장하지만,
        query = """
        -- users 테이블
        CREATE TABLE IF NOT EXISTS users (
            id SERIAL PRIMARY KEY,
            name TEXT NOT NULL,
            rank TEXT,
            dept TEXT
        );

        -- document 테이블
        CREATE TABLE IF NOT EXISTS document (
            id SERIAL PRIMARY KEY,
            doc_name TEXT NOT NULL,
            content TEXT,                 -- 원본 전체 마크다운 또는 텍스트
            doc_type TEXT,                -- 문서 타입 (.pdf, .docx 등)
            version TEXT DEFAULT '1.0',   -- 문서 버전
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        -- chunk 테이블
        CREATE TABLE IF NOT EXISTS chunk (
            id SERIAL PRIMARY KEY,
            clause TEXT,                  -- 조항 번호 (ex 1.1, 5.1.2)
            content TEXT NOT NULL,        -- 청크 내용
            metadata JSONB,               -- 청크 메타데이터 (헤더, 섹션 등)
            document_id INTEGER REFERENCES document(id) ON DELETE CASCADE
        );

        -- memory 테이블
        CREATE TABLE IF NOT EXISTS memory (
            id SERIAL PRIMARY KEY,
            answer TEXT,
            question TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            users_id INTEGER REFERENCES users(id) ON DELETE SET NULL
        );

        -- 인덱스 생성
        CREATE INDEX IF NOT EXISTS idx_chunk_document_id ON chunk(document_id);
        CREATE INDEX IF NOT EXISTS idx_chunk_clause ON chunk(clause);
        CREATE INDEX IF NOT EXISTS idx_chunk_metadata ON chunk USING GIN (metadata);
        CREATE INDEX IF NOT EXISTS idx_memory_users_id ON memory(users_id);
        CREATE INDEX IF NOT EXISTS idx_document_doc_name ON document(doc_name);
        """
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(query)
                    conn.commit()
            print("✅ [SQLStore] PostgreSQL 테이블이 준비되었습니다 (document, chunk, users, memory).")
        except Exception as e:
            print(f"❌ [SQLStore] DB 초기화 실패: {e}")

    def save_document(
        self,
        doc_name: str,
        content: str,
        doc_type: str = None,
        version: str = "1.0"
    ) -> Optional[int]:
        """문서 정보를 저장합니다."""
        insert_query = """
            INSERT INTO document (doc_name, content, doc_type, version)
            VALUES (%s, %s, %s, %s)
            RETURNING id;
        """

        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(insert_query, (doc_name, content, doc_type, version))
                    doc_id = cur.fetchone()[0]
                    conn.commit()
            print(f"✅ [SQLStore] 문서 저장 성공: {doc_name} v{version} (ID: {doc_id})")
            return doc_id
        except Exception as e:
            print(f"❌ [SQLStore] 문서 저장 실패: {e}")
            return None

    def save_chunk(
        self,
        document_id: int,
        clause: str,
        content: str,
        metadata: Dict = None
    ) -> Optional[int]:
        """청크를 저장합니다."""
        insert_query = """
            INSERT INTO chunk (clause, content, metadata, document_id)
            VALUES (%s, %s, %s, %s)
            RETURNING id;
        """

        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(insert_query, (
                        clause,
                        content,
                        json.dumps(metadata or {}),
                        document_id
                    ))
                    chunk_id = cur.fetchone()[0]
                    conn.commit()
            return chunk_id
        except Exception as e:
            print(f"❌ [SQLStore] 청크 저장 실패: {e}")
            return None

    def save_chunks_batch(
        self,
        document_id: int,
        chunks: List[Dict]
    ):
        """여러 청크를 일괄 저장합니다."""
        insert_query = """
            INSERT INTO chunk (clause, content, metadata, document_id)
            VALUES (%s, %s, %s, %s);
        """

        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    for chunk in chunks:
                        cur.execute(insert_query, (
                            chunk.get('clause'),
                            chunk.get('content'),
                            json.dumps(chunk.get('metadata', {})),
                            document_id
                        ))
                    conn.commit()
            print(f"✅ [SQLStore] {len(chunks)}개 청크 저장 성공 (document_id: {document_id})")
        except Exception as e:
            print(f"❌ [SQLStore] 청크 일괄 저장 실패: {e}")

    def get_document_by_id(self, document_id: int) -> Optional[Dict]:
        """문서 ID로 문서 조회"""
        query = "SELECT id, doc_name, content, doc_type, version, created_at FROM document WHERE id = %s"
        try:
            with self._get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(query, (document_id,))
                    return cur.fetchone()
        except Exception as e:
            return None

    def get_document_by_name(self, doc_name: str, version: str = None) -> Optional[Dict]:
        """문서명으로 문서 조회 (버전 미지정 시 최신 버전)"""
        if version:
            query = "SELECT id, doc_name, content, doc_type, version, created_at FROM document WHERE doc_name = %s AND version = %s"
            params = (doc_name, version)
        else:
            # 최신 버전 (created_at 기준 내림차순)
            query = "SELECT id, doc_name, content, doc_type, version, created_at FROM document WHERE doc_name = %s ORDER BY created_at DESC LIMIT 1"
            params = (doc_name,)

        try:
            with self._get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(query, params)
                    return cur.fetchone()
        except Exception as e:
            return None

    def get_chunks_by_document(self, document_id: int) -> List[Dict]:
        """특정 문서의 모든 청크 조회"""
        query = "SELECT id, clause, content, metadata FROM chunk WHERE document_id = %s ORDER BY id"
        try:
            with self._get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(query, (document_id,))
                    return cur.fetchall()
        except Exception:
            return []

    def list_documents(self) -> List[Dict]:
        """모든 문서 목록 조회"""
        query = "SELECT id, doc_name, doc_type, version, created_at FROM document ORDER BY created_at DESC"
        try:
            with self._get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(query)
                    return cur.fetchall()
        except Exception as e:
            print(f"❌ [SQLStore] 목록 조회 실패: {e}")
            return []

    # Users 테이블 관련 메서드
    def save_user(self, name: str, rank: str = None, dept: str = None) -> Optional[int]:
        """사용자 저장"""
        insert_query = "INSERT INTO users (name, rank, dept) VALUES (%s, %s, %s) RETURNING id;"
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(insert_query, (name, rank, dept))
                    user_id = cur.fetchone()[0]
                    conn.commit()
            return user_id
        except Exception as e:
            print(f"❌ [SQLStore] 사용자 저장 실패: {e}")
            return None

    def get_user(self, user_id: int) -> Optional[Dict]:
        """사용자 조회"""
        query = "SELECT id, name, rank, dept FROM users WHERE id = %s"
        try:
            with self._get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(query, (user_id,))
                    return cur.fetchone()
        except Exception:
            return None

    # Memory 테이블 관련 메서드
    def save_memory(self, question: str, answer: str, users_id: int = None) -> Optional[int]:
        """대화 기록 저장"""
        insert_query = "INSERT INTO memory (question, answer, users_id) VALUES (%s, %s, %s) RETURNING id;"
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(insert_query, (question, answer, users_id))
                    memory_id = cur.fetchone()[0]
                    conn.commit()
            return memory_id
        except Exception as e:
            print(f"❌ [SQLStore] 대화 기록 저장 실패: {e}")
            return None

    def get_memory_by_user(self, users_id: int, limit: int = 10) -> List[Dict]:
        """사용자별 대화 기록 조회"""
        query = """
            SELECT id, question, answer, created_at
            FROM memory
            WHERE users_id = %s
            ORDER BY created_at DESC
            LIMIT %s
        """
        try:
            with self._get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(query, (users_id, limit))
                    return cur.fetchall()
        except Exception:
            return []

if __name__ == "__main__":
    # 테스트
    store = SQLStore()
    store.init_db()

    # 사용자 생성 테스트
    user_id = store.save_user("홍길동", "사원", "품질관리팀")
    print(f"생성된 사용자 ID: {user_id}")

    # 문서 생성 테스트
    doc_id = store.save_document(
        doc_name="EQ-SOP-00010",
        content="# 품질관리기준서\n\n## 1. 목적\n본 기준서는...",
        doc_type=".md",
        version="1.0"
    )
    print(f"생성된 문서 ID: {doc_id}")

    # 청크 생성 테스트
    if doc_id:
        chunk_id = store.save_chunk(
            document_id=doc_id,
            clause="1.1",
            content="본 기준서는 품질관리기준서의 작성, 검토, 승인에 관한 기준을 정한다.",
            metadata={"section": "목적", "H2": "1. 목적"}
        )
        print(f"생성된 청크 ID: {chunk_id}")

    # 대화 기록 저장 테스트
    if user_id:
        memory_id = store.save_memory(
            question="품질관리기준서는 어떻게 작성하나요?",
            answer="품질관리기준서는 작성, 검토, 승인 절차를 따릅니다.",
            users_id=user_id
        )
        print(f"생성된 대화 기록 ID: {memory_id}")
