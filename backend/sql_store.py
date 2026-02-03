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
        # sop_id의 UNIQUE 제약조건을 제거하고 (sop_id, version) 복합 유니크를 권장하지만,
        # 기존 호환성을 위해 유연하게 인덱스만 생성합니다.
        query = """
        CREATE TABLE IF NOT EXISTS sop_documents (
            id SERIAL PRIMARY KEY,
            sop_id TEXT,                  -- SOP 고유 번호 (중복 허용, 버전별 저장)
            title TEXT,                   -- 문서 제목
            markdown_content TEXT,        -- 원본 전체 마크다운 (요약용)
            pdf_binary BYTEA,             -- 원본 PDF 데이터
            doc_metadata JSONB,           -- 버전, 시행일, 부서 등 (문서 레밸)
            stats JSONB DEFAULT '{"hit_count": 0, "last_accessed": null}'::jsonb,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        -- 검색 속도를 위한 인덱스
        CREATE INDEX IF NOT EXISTS idx_sop_id ON sop_documents(sop_id);
        CREATE INDEX IF NOT EXISTS idx_doc_metadata ON sop_documents USING GIN (doc_metadata);
        """
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(query)
                    conn.commit()
            print("✅ [SQLStore] PostgreSQL 테이블이 준비되었습니다 (버전 관리 지원).")
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
        """문서 전체 정보를 저장합니다. (버전별로 신규 행 추가)"""
        # 기존: ON CONFLICT UPDATE -> 변경: 동일 버전이 있으면 업데이트, 없으면 INSERT
        # 여기서는 간단하게 항상 INSERT 하되, 실제 운영환경에서는 버전 체크 로직이 필요합니다.
        # 편의상, doc_metadata 내의 version 을 확인하여 중복 체크
        
        version = doc_metadata.get("version", "1.0") if doc_metadata else "1.0"
        
        check_query = "SELECT id FROM sop_documents WHERE sop_id = %s AND doc_metadata->>'version' = %s"
        update_query = """
            UPDATE sop_documents 
            SET title = %s, markdown_content = %s, pdf_binary = %s, doc_metadata = %s, updated_at = CURRENT_TIMESTAMP
            WHERE id = %s
            RETURNING id;
        """
        insert_query = """
            INSERT INTO sop_documents (sop_id, title, markdown_content, pdf_binary, doc_metadata, updated_at)
            VALUES (%s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
            RETURNING id;
        """
        
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    # 중복 체크
                    cur.execute(check_query, (sop_id, version))
                    exist = cur.fetchone()
                    
                    if exist:
                        doc_id = exist[0]
                        cur.execute(update_query, (
                            title,
                            markdown_content,
                            psycopg2.Binary(pdf_binary) if pdf_binary else None,
                            json.dumps(doc_metadata or {}),
                            doc_id
                        ))
                    else:
                        cur.execute(insert_query, (
                            sop_id,
                            title,
                            markdown_content,
                            psycopg2.Binary(pdf_binary) if pdf_binary else None,
                            json.dumps(doc_metadata or {})
                        ))
                        doc_id = cur.fetchone()[0]
                    
                    conn.commit()
            print(f"✅ [SQLStore] 문서 저장 성공: {sop_id} v{version} (ID: {doc_id})")
            return doc_id
        except Exception as e:
            print(f"❌ [SQLStore] 문서 저장 실패: {e}")
            return None

    def get_document_by_id(self, sop_id: str, version: str = None) -> Optional[Dict]:
        """SOP ID로 문서 조회 (버전 미지정 시 최신 버전)"""
        if version:
            query = "SELECT sop_id, title, markdown_content, doc_metadata FROM sop_documents WHERE sop_id = %s AND doc_metadata->>'version' = %s"
            params = (sop_id, version)
        else:
            # 최신 버전 (updated_at 기준 내림차순)
            query = "SELECT sop_id, title, markdown_content, doc_metadata FROM sop_documents WHERE sop_id = %s ORDER BY updated_at DESC LIMIT 1"
            params = (sop_id,)

        # 조회수 업데이트 (최신 1건에 대해서만)
        update_stats = """
        UPDATE sop_documents 
        SET stats = jsonb_set(stats, '{hit_count}', ((stats->>'hit_count')::int + 1)::text::jsonb)
        WHERE sop_id = %s
        """
        try:
            with self._get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(query, params)
                    doc = cur.fetchone()
                    if doc:
                        # 통계 업데이트는 sop_id 기준으로 전체 적용 (단순화)
                        cur.execute(update_stats, (sop_id,))
                        conn.commit()
                    return doc
        except Exception as e:
            # print(f"❌ [SQLStore] 문서 조회 실패: {e}") 
            return None
            
    def get_document_versions(self, sop_id: str) -> List[Dict]:
        """특정 SOP ID의 모든 버전 목록 조회"""
        query = "SELECT sop_id, title, doc_metadata, created_at FROM sop_documents WHERE sop_id = %s ORDER BY created_at DESC"
        try:
            with self._get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(query, (sop_id,))
                    return cur.fetchall()
        except Exception:
            return []

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
