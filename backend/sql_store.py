import psycopg2
from psycopg2.extras import RealDictCursor
import json
import os
from typing import List, Dict, Any, Optional

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
        query = """
        CREATE TABLE IF NOT EXISTS sop_documents (
            id SERIAL PRIMARY KEY,
            sop_id TEXT UNIQUE,          -- SOP 고유 번호 (EQ-SOP-001)
            title TEXT,                   -- 문서 제목
            markdown_content TEXT,        -- 원본 전체 마크다운 (요약용)
            pdf_binary BYTEA,             -- 원본 PDF 데이터
            doc_metadata JSONB,           -- 버전, 시행일, 부서 등 (문서 레밸)
            stats JSONB DEFAULT '{"hit_count": 0, "last_accessed": null}'::jsonb,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE INDEX IF NOT EXISTS idx_sop_id ON sop_documents(sop_id);
        CREATE INDEX IF NOT EXISTS idx_doc_metadata ON sop_documents USING GIN (doc_metadata);
        """
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(query)
                    conn.commit()
            print("✅ [SQLStore] PostgreSQL 테이블이 준비되었습니다.")
        except Exception as e:
            print(f"❌ [SQLStore] DB 초기화 실패: {e}")

    def save_document(
        self, 
        sop_id: str, 
        title: str, 
        markdown_content: str, 
        pdf_binary: bytes = None,
        doc_metadata: Dict = None
    ):
        """문서 전체 정보를 저장 또는 업데이트합니다."""
        upsert_query = """
        INSERT INTO sop_documents (sop_id, title, markdown_content, pdf_binary, doc_metadata, updated_at)
        VALUES (%s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
        ON CONFLICT (sop_id) DO UPDATE SET
            title = EXCLUDED.title,
            markdown_content = EXCLUDED.markdown_content,
            pdf_binary = COALESCE(EXCLUDED.pdf_binary, sop_documents.pdf_binary),
            doc_metadata = EXCLUDED.doc_metadata,
            updated_at = CURRENT_TIMESTAMP
        RETURNING id;
        """
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(upsert_query, (
                        sop_id,
                        title,
                        markdown_content,
                        psycopg2.Binary(pdf_binary) if pdf_binary else None,
                        json.dumps(doc_metadata or {})
                    ))
                    doc_id = cur.fetchone()[0]
                    conn.commit()
            print(f"✅ [SQLStore] 문서 저장 성공: {sop_id} (ID: {doc_id})")
            return doc_id
        except Exception as e:
            print(f"❌ [SQLStore] 문서 저장 실패: {e}")
            return None

    def get_document_by_id(self, sop_id: str) -> Optional[Dict]:
        """SOP ID로 문서 전체 정보 및 원본 텍스트를 조회합니다."""
        query = "SELECT sop_id, title, markdown_content, doc_metadata FROM sop_documents WHERE sop_id = %s"
        # 조회수 업데이트
        update_stats = """
        UPDATE sop_documents 
        SET stats = jsonb_set(stats, '{hit_count}', ((stats->>'hit_count')::int + 1)::text::jsonb)
        WHERE sop_id = %s
        """
        try:
            with self._get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(query, (sop_id,))
                    doc = cur.fetchone()
                    if doc:
                        cur.execute(update_stats, (sop_id,))
                        conn.commit()
                    return doc
        except Exception as e:
            # print(f"❌ [SQLStore] 문서 조회 실패: {e}") # 로그 과다 방지
            return None

    def list_documents(self, department: str = None) -> List[Dict]:
        """문서 목록을 조회합니다. (필터링 지원)"""
        if department:
            query = "SELECT sop_id, title, doc_metadata FROM sop_documents WHERE doc_metadata->>'department' = %s"
            params = (department,)
        else:
            query = "SELECT sop_id, title, doc_metadata FROM sop_documents"
            params = ()
            
        try:
            with self._get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(query, params)
                    return cur.fetchall()
        except Exception as e:
            print(f"❌ [SQLStore] 목록 조회 실패: {e}")
            return []

if __name__ == "__main__":
    store = SQLStore()
    store.init_db()
