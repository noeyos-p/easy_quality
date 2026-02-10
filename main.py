"""
RAG 챗봇 API v14.0 + Agent (OpenAI)

 v14.0 변경사항:
- LLM 백엔드 변경: Z.AI → OpenAI GPT-4o-mini
- 에이전트 시스템 통합 (모든 서브 에이전트 OpenAI 사용)
- LLM as a Judge 평가 시스템 (RDB 검증 포함)
- LangSmith 추적 지원 및 최적화
"""

#  .env 파일 자동 로드 (다른 import보다 먼저!)
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from contextlib import asynccontextmanager
import torch
import time
import uuid
import re

from backend.sql_store import SQLStore
sql_store = SQLStore()
# sql_store.init_db()  #  main()으로 이동하여 중복 호출 방지

# RAG 모듈 - 레거시 (폴백용)
# RAG 모듈 - 레거시 (폴백용) 제거됨
# LangGraph 파이프라인이 전적으로 처리

from sentence_transformers import SentenceTransformer
from backend import vector_store
# from backend.prompt import build_rag_prompt, build_chunk_prompt (제거됨)
from backend.llm import (
    get_llm_response,
    ZaiLLM,
    OllamaLLM,
    analyze_search_results,
    HUGGINGFACE_MODELS,
)

#  Document pipeline
try:
    from backend.document_pipeline import process_document
    from dataclasses import dataclass

    @dataclass
    class Chunk:
        text: str
        metadata: dict
        index: int = 0

    LANGGRAPH_AVAILABLE = True
    print(" Document pipeline 사용 가능")
except ImportError as e:
    LANGGRAPH_AVAILABLE = False
    print(f" Document pipeline 사용 불가: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    yield
    # Shutdown
    print("\n 서버 종료 중...")
    vector_store.close_client()
    if _graph_store:
        _graph_store.close()
        print(" Neo4j 연결 종료됨")

app = FastAPI(title="RAG Chatbot API", version="9.2.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ═══════════════════════════════════════════════════════════════════════════
# 설정
# ═══════════════════════════════════════════════════════════════════════════

DEFAULT_CHUNK_SIZE = 500
DEFAULT_OVERLAP = 50
DEFAULT_CHUNK_METHOD = "article"
DEFAULT_N_RESULTS = 7  #  5 -> 7 상향
DEFAULT_SIMILARITY_THRESHOLD = 0.30  #  0.35 -> 0.30 (더 많은 맥락 확보)
USE_LANGGRAPH = True  #  LangGraph 파이프라인 사용 여부

PRESET_MODELS = {
    "multilingual-e5-small": "intfloat/multilingual-e5-small",
}


device = "cuda" if torch.cuda.is_available() else "cpu"

# 대화 히스토리 저장 (메모리)
chat_histories: Dict[str, List[Dict]] = {}

# Neo4j 그래프 스토어 (싱글톤)
_graph_store = None


def get_graph_store():
    """Neo4j 그래프 스토어 싱글톤"""
    global _graph_store
    if _graph_store is None:
        from backend.graph_store import Neo4jGraphStore
        _graph_store = Neo4jGraphStore()
        _graph_store.connect()
    return _graph_store


# ═══════════════════════════════════════════════════════════════════════════
# Pydantic 모델
# ═══════════════════════════════════════════════════════════════════════════

class SearchRequest(BaseModel):
    query: str
    collection: str = "documents"
    n_results: int = DEFAULT_N_RESULTS
    model: str = "multilingual-e5-small"
    filter_doc: Optional[str] = None
    similarity_threshold: Optional[float] = None


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    collection: str = "documents"
    n_results: int = DEFAULT_N_RESULTS
    embedding_model: str = "multilingual-e5-small"
    llm_model: str = "gpt-4o-mini"
    llm_backend: str = "openai"
    filter_doc: Optional[str] = None
    similarity_threshold: Optional[float] = None


class AskRequest(BaseModel):
    query: str
    collection: str = "documents"
    n_results: int = DEFAULT_N_RESULTS
    embedding_model: str = "multilingual-e5-small"
    llm_model: str = "gpt-4o-mini"
    llm_backend: str = "openai"  #  기본값 openai로 변경
    temperature: float = 0.7
    filter_doc: Optional[str] = None
    language: str = "ko"
    max_tokens: int = 512
    similarity_threshold: Optional[float] = None
    include_sources: bool = True


class LLMRequest(BaseModel):
    prompt: str
    model: str = "qwen2.5:3b"
    backend: str = "ollama"
    max_tokens: int = 256
    temperature: float = 0.1


class DeleteDocRequest(BaseModel):
    doc_name: str
    collection: str = "documents"
    delete_from_neo4j: bool = True  #  Neo4j에서도 삭제


# ═══════════════════════════════════════════════════════════════════════════
# 헬퍼 함수
# ═══════════════════════════════════════════════════════════════════════════

def resolve_model_path(model: str) -> str:
    """모델 프리셋 → 전체 경로"""
    return PRESET_MODELS.get(model, model)


def format_context(results: List[Dict]) -> str:
    """검색 결과 → 컨텍스트 문자열 (메타데이터 포함)"""
    context_parts = []
    
    for i, r in enumerate(results, 1):
        meta = r.get("metadata", {})
        text = r.get("text", "")
        similarity = r.get("similarity", 0)
        
        #  v9.2: 개선된 출처 표시
        doc_id = meta.get("doc_id", "")
        section_path = meta.get("section_path", "")
        page = meta.get("page", "")
        article_num = meta.get("article_num", "")
        
        # 출처 헤더 구성
        source_parts = []
        if doc_id:
            source_parts.append(f"[{doc_id}]")
        if section_path:
            source_parts.append(f"> {section_path}")
        if page:
            source_parts.append(f"(p.{page})")
        if similarity:
            source_parts.append(f"관련도: {similarity:.0%}")
        
        source_header = " ".join(source_parts) if source_parts else f"[문서 {i}]"
        
        context_parts.append(f"{source_header}\n{text}")
    
    return "\n\n---\n\n".join(context_parts)


# ═══════════════════════════════════════════════════════════════════════════
# API 엔드포인트 - 기본
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/")
def root():
    return {
        "message": "RAG Chatbot API v9.2",
        "features": [
            "LangGraph 파이프라인",
            "페이지 번호 추적",
            "Parent-Child 계층",
            "Question 추적 (Neo4j)",
            "Weaviate + Neo4j 동기화 삭제"
        ],
        "endpoints": {
            "upload": "/rag/upload",
            "search": "/rag/search",
            "chat": "/chat",
            "ask": "/rag/ask",
            "graph": "/graph/*"
        },
        "langgraph_enabled": LANGGRAPH_AVAILABLE and USE_LANGGRAPH
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "cuda": torch.cuda.is_available(),
        "device": device,
        "ollama": OllamaLLM.is_available(),
        "langgraph": LANGGRAPH_AVAILABLE
    }


@app.get("/models/embedding")
def list_embedding_models():
    return {
        "presets": PRESET_MODELS,
        "specs": vector_store.EMBEDDING_MODEL_SPECS,
        "compatible": vector_store.filter_compatible_models()
    }


@app.get("/models/llm")
def list_llm_models():
    available_ollama = []
    if OllamaLLM.is_available():
        available_ollama = OllamaLLM.list_models()
    return {
        "ollama": {"presets": OLLAMA_MODELS, "available": available_ollama},
        "huggingface": HUGGINGFACE_MODELS
    }


# ═══════════════════════════════════════════════════════════════════════════
#  API 엔드포인트 - 업로드 (LangGraph v9.2)
# ═══════════════════════════════════════════════════════════════════════════

@app.post("/rag/upload")
async def upload_document(
    file: UploadFile = File(...),
    collection: str = Form("documents"),
    chunk_size: int = Form(DEFAULT_CHUNK_SIZE),
    chunk_method: str = Form(DEFAULT_CHUNK_METHOD),
    model: str = Form("multilingual-e5-small"),
    overlap: int = Form(DEFAULT_OVERLAP),
    use_langgraph: bool = Form(True),  #  LangGraph 사용 여부
    use_llm_metadata: bool = Form(True),  #  LLM 메타데이터 추출 사용 여부
    version: Optional[str] = Form(None), # 사용자가 직접 지정하는 버전
):
    """
    문서 업로드 (LangGraph v9.2 파이프라인)
    
    - ChromaDB에 벡터 저장
    - Neo4j에 그래프 저장
    - 페이지 번호, Parent-Child 계층 메타데이터 포함
    """
    start_time = time.time()
    
    try:
        content = await file.read()
        filename = file.filename
        
        print(f"\n{'='*70}")
        print(f"문서 업로드: {filename}")
        print(f"{'='*70}\n")

        # ========================================
        # 문서 파싱
        # ========================================
        print(f"[1단계] 문서 파싱")
        print(f"  파이프라인: PDF 조항 v2.0")
        print(f"  LLM 메타데이터: {'🟢 활성' if use_llm_metadata else '비활성'}")
        if use_llm_metadata:
            print(f"  LLM 모델: gpt-4o-mini")
        print()

        model_path = resolve_model_path(model)
        embed_model = SentenceTransformer(model_path)

        result = process_document(
            file_path=filename,
            content=content,
            use_llm_metadata=use_llm_metadata,
            embed_model=embed_model
        )

        if not result.get("success"):
            errors = result.get("errors", ["알 수 없는 오류"])
            raise HTTPException(400, f"🔴 문서 처리 실패: {errors}")

        chunks_data = result["chunks"]
        if not chunks_data:
            raise HTTPException(400, "🔴 텍스트 추출 실패")

        from dataclasses import dataclass
        @dataclass
        class Chunk:
            text: str
            metadata: dict
            index: int = 0

        chunks = [Chunk(text=c["text"], metadata=c["metadata"], index=c["index"]) for c in chunks_data]
        doc_id = result.get("doc_id")
        doc_title = result.get("doc_title")
        pipeline_version = "pdf-clause-v2.0"

        print(f"  🟢 파싱 완료")
        print(f"     • ID: {doc_id}")
        print(f"     • 제목: {doc_title}")
        print(f"     • 조항: {result.get('total_clauses')}개")
        print(f"     • 청크: {len(chunks)}개\n")
        
        # ========================================
        # Weaviate 벡터 저장
        # ========================================
        print(f"[2단계] Weaviate 벡터 저장")

        texts = [c.text for c in chunks]
        metadatas = [
            {
                **c.metadata,
                "chunk_method": chunk_method,
                "model": model,
                "pipeline_version": pipeline_version,
            }
            for c in chunks
        ]

        vector_store.add_documents(
            texts=texts,
            metadatas=metadatas,
            collection_name=collection,
            model_name=model_path
        )
        print(f"  🟢 저장 완료: {len(chunks)}개 청크\n")
        
        # ========================================
        # PostgreSQL 문서 저장
        # ========================================
        print(f"[3단계] PostgreSQL 저장")

        try:
            full_markdown = "\n\n".join([c.text for c in chunks])

            # 파이프라인에서 추출된 버전 또는 사용자 입력 버전 결정
            final_version = version or result.get("version", "1.0")
            
            if final_version != "1.0":
                print(f"     [추출] 최종 결정된 버전: {final_version}")

            doc_id_db = sql_store.save_document(
                doc_name=doc_id,
                content=full_markdown,
                doc_type=filename.split('.')[-1] if '.' in filename else None,
                version=final_version
            )

            if doc_id_db and chunks:
                batch_chunks = [
                    {
                        "clause": c.metadata.get("clause_id"),
                        "content": c.text,
                        "metadata": c.metadata
                    }
                    for c in chunks
                ]
                sql_store.save_chunks_batch(doc_id_db, batch_chunks)
                print(f"  🟢 저장 완료: 문서 + {len(chunks)}개 청크\n")
            else:
                print(f"  🔴 저장 실패: DB 저장에 실패했습니다 (ID 생성 불가)\n")
        except Exception as sql_err:
            print(f"  🔴 저장 실패: {sql_err}\n")

        # ========================================
        # Neo4j 그래프 저장
        # ========================================
        print(f"[4단계] Neo4j 그래프 저장")
        graph_uploaded = False
        graph_sections = 0

        try:
            from backend.graph_store import Neo4jGraphStore

            graph = get_graph_store()
            if graph.test_connection():
                _upload_to_neo4j_from_pipeline(graph, result, filename)
                graph_uploaded = True
                stats = graph.get_graph_stats()
                graph_sections = stats.get("sections", 0)
                print(f"  🟢 저장 완료: {graph_sections}개 섹션\n")
        except Exception as graph_error:
            # [디버그 로그 보강] 연결 실패 시 구체적인 에러 메시지 출력
            print(f"  🔴 Neo4j 연결 실패: {graph_error}")
            import traceback
            traceback.print_exc()
            print(f"  ⚠ 그래프 연동을 건너뛰고 계속 진행합니다.\n")
        
        # ========================================
        # 완료
        # ========================================
        elapsed = round(time.time() - start_time, 2)

        print(f"{'='*70}")
        print(f"🟢 업로드 완료 ({elapsed}초)")
        print(f"{'='*70}\n")

        return {
            "success": True,
            "filename": filename,
            "doc_id": doc_id,
            "doc_title": doc_title,
            "version": final_version,
            "chunks": len(chunks),
            "total_clauses": result.get("total_clauses"),
            "chunk_method": chunk_method,
            "pipeline_version": pipeline_version,
            "graph_uploaded": graph_uploaded,
            "elapsed_seconds": elapsed,
            "sample_metadata": metadatas[0] if metadatas else {},
        }

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(500, f"🔴 실패: {str(e)}")


def _upload_to_neo4j_from_pipeline(graph, result: dict, filename: str):
    """새 파이프라인 결과를 Neo4j에 업로드 (간소화)"""
    from backend.graph_store import upload_document_to_graph
    upload_document_to_graph(graph, result, filename)


# ═══════════════════════════════════════════════════════════════════════════
# API 엔드포인트 - 검색
# ═══════════════════════════════════════════════════════════════════════════

# /rag/search 엔드포인트 제거됨 (Agent가 내부 수행)


# /rag/search/advanced 엔드포인트 제거됨


# ═══════════════════════════════════════════════════════════════════════════
# API 엔드포인트 - 챗봇
# ═══════════════════════════════════════════════════════════════════════════

@app.post("/chat")
def chat(request: ChatRequest):
    """
    Main Agent Chat Endpoint
    - Manual RAG 로직 제거됨
    - 오직 Agent Orchestrator를 통해서만 답변
    """
    print(f" [Agent] 요청 수신: {request.message}")
    
    try:
        # Agent 실행
        # llm.py 업데이트에 따라 model_name 파라미터 등을 적절히 전달
        init_agent_tools(vector_store, get_graph_store(), sql_store)
        
        response = run_agent(
            query=request.message,
            session_id=request.session_id or str(uuid.uuid4()),
            model_name=request.llm_model or "gpt-4o-mini"
        )

        answer = response.get("answer")

        # LLM as a Judge 평가
        evaluation_scores = None

        # 에러 메시지 패턴 감지
        error_patterns = ["오류가 발생", "에러", "실패", "Error", "Exception", "찾을 수 없", "준비하지 못", "로딩 에러"]
        is_error_message = any(pattern in answer for pattern in error_patterns)

        try:
            from backend.evaluation import AgentEvaluator

            # 평가 생략 조건
            if len(answer) < 20:
                print("평가 생략: 답변이 너무 짧음")
            elif is_error_message:
                print("평가 생략: 에러 메시지")
            else:
                # 평가 실행 (RDB 검증 필수!)
                evaluator = AgentEvaluator(
                    judge_model="gpt-4o-mini",
                    sql_store=sql_store  # ✅ RDB 검증을 위해 필수 전달
                )

                # context 추출 (agent_log에서)
                context = response.get("agent_log", {}).get("context", "")
                if isinstance(context, list):
                    context = "\n\n".join(context)

                evaluation_scores = evaluator.evaluate_single(
                    question=request.message,
                    answer=answer,
                    context=context,
                    metrics=["faithfulness", "groundness", "relevancy", "correctness"]
                )

                # 로그 출력
                if evaluation_scores:
                    print(f"\n{'='*60}")
                    print(f"평가 결과 (평균: {evaluation_scores.get('average_score', 0)}/5)")
                    print(f"{'='*60}")
                    for metric, result in evaluation_scores.items():
                        # average_score는 건너뜀 (float이므로 .get() 메서드 없음)
                        if metric == "average_score":
                            continue

                        score = result.get("score", 0)
                        reasoning = result.get("reasoning", "")
                        print(f"\n[{metric.upper()}]")
                        print(f"  점수: {score}/5")
                        print(f"  이유: {reasoning}")

                        # RDB 검증 결과 출력
                        if "rdb_verification" in result:
                            rdb = result["rdb_verification"]
                            print(f"  📊 RDB 검증: 정확도 {rdb.get('accuracy_rate', 0)}% ({rdb.get('verified_citations', 0)}/{rdb.get('total_citations', 0)})")
                    print(f"{'='*60}\n")

        except ImportError:
            print("평가 모듈 사용 불가 (선택적 기능)")
        except Exception as eval_error:
            print(f"평가 실행 실패 (계속 진행): {eval_error}")
            evaluation_scores = None

        return {
            "session_id": request.session_id,
            "answer": answer,
            "sources": [],
            "agent_log": response,
            "evaluation_scores": evaluation_scores
        }
    except Exception as e:
        print(f" [Agent] 에러: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(500, str(e))



@app.get("/chat/history/{session_id}")
def get_chat_history(session_id: str):
    """대화 히스토리 조회"""
    history = chat_histories.get(session_id, [])
    return {"session_id": session_id, "history": history, "count": len(history)}


@app.delete("/chat/history/{session_id}")
def clear_chat_history(session_id: str):
    """대화 히스토리 삭제"""
    if session_id in chat_histories:
        del chat_histories[session_id]
        return {"success": True, "message": f"세션 {session_id} 삭제됨"}
    return {"success": False, "message": "세션을 찾을 수 없음"}


# ═══════════════════════════════════════════════════════════════════════════
# API 엔드포인트 - LLM
# ═══════════════════════════════════════════════════════════════════════════

@app.post("/llm/generate")
def generate_llm(request: LLMRequest):
    """LLM 직접 호출"""
    try:
        response = get_llm_response(
            prompt=request.prompt,
            llm_model=request.model,
            llm_backend=request.backend,
            max_tokens=request.max_tokens,
            temperature=request.temperature
        )
        return {"response": response, "model": request.model, "backend": request.backend}
    except Exception as e:
        raise HTTPException(500, f"LLM 호출 실패: {str(e)}")


# ═══════════════════════════════════════════════════════════════════════════
# API 엔드포인트 - 문서 관리
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/rag/documents")
def list_documents(collection: str = "documents"):
    """문서 목록"""
    docs = vector_store.list_documents(collection)
    return {"documents": docs, "collection": collection}


@app.delete("/rag/document")
def delete_document(request: DeleteDocRequest):
    """
     문서 삭제 (Weaviate + Neo4j 동시 삭제)
    """
    result = {"chromadb": None, "neo4j": None}
    
    # 1. Weaviate 삭제
    chroma_result = vector_store.delete_by_doc_name(
        doc_name=request.doc_name,
        collection_name=request.collection
    )
    result["weaviate"] = chroma_result
    
    # 2. Neo4j 삭제 (옵션)
    if request.delete_from_neo4j:
        try:
            graph = get_graph_store()
            if graph.test_connection():
                # doc_name에서 doc_id 추출 시도
                import re
                sop_match = re.search(r'(EQ-SOP-\d+)', request.doc_name, re.IGNORECASE)
                if sop_match:
                    doc_id = sop_match.group(1).upper()
                    neo4j_result = graph.delete_document(doc_id)
                    result["neo4j"] = {"success": True, "doc_id": doc_id, "deleted": neo4j_result}
                else:
                    result["neo4j"] = {"success": False, "message": "SOP ID를 추출할 수 없음"}
        except Exception as e:
            result["neo4j"] = {"success": False, "error": str(e)}
    
    # 전체 성공 여부
    success = chroma_result.get("success", False)
    
    return {
        "success": success,
        "doc_name": request.doc_name,
        "details": result
    }


@app.get("/rag/collections")
def list_collections():
    """컬렉션 목록"""
    collections = vector_store.list_collections()
    return {"collections": [vector_store.get_collection_info(name) for name in collections]}


@app.delete("/rag/collection/{collection_name}")
def delete_collection(collection_name: str):
    """컬렉션 삭제"""
    return vector_store.delete_all(collection_name)


@app.get("/rag/supported-formats")
def get_supported_formats():
    """지원 포맷"""
    return {"supported_extensions": get_supported_extensions()}


@app.get("/rag/chunk-methods")
def get_chunk_methods():
    """청킹 방법"""
    return {"methods": get_available_methods()}


# ═══════════════════════════════════════════════════════════════════════════
# API 엔드포인트 - Neo4j 그래프
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/graph/status")
def graph_status():
    """Neo4j 연결 상태"""
    try:
        graph = get_graph_store()
        connected = graph.test_connection()
        stats = graph.get_graph_stats() if connected else {}
        return {"connected": connected, "stats": stats}
    except Exception as e:
        return {"connected": False, "error": str(e)}


@app.post("/graph/init")
def graph_init():
    """Neo4j 스키마 초기화"""
    try:
        graph = get_graph_store()
        graph.init_schema()
        return {"success": True, "message": "스키마 초기화 완료"}
    except Exception as e:
        raise HTTPException(500, f"스키마 초기화 실패: {str(e)}")


@app.delete("/graph/clear")
def graph_clear():
    """Neo4j 모든 데이터 삭제"""
    try:
        graph = get_graph_store()
        graph.clear_all()
        return {"success": True, "message": "모든 데이터 삭제 완료"}
    except Exception as e:
        raise HTTPException(500, f"데이터 삭제 실패: {str(e)}")


@app.post("/graph/upload")
async def graph_upload_document(
    file: UploadFile = File(...),
    use_langgraph: bool = Form(True)
):
    """문서를 Neo4j 그래프로만 업로드"""
    try:
        content = await file.read()
        filename = file.filename
        
        if not LANGGRAPH_AVAILABLE:
            raise HTTPException(500, "LangGraph 모듈이 필요합니다.")
            
        result = process_document(filename, content, debug=True)
        if not result.get("success"):
            raise HTTPException(400, f"처리 실패: {result.get('errors')}")
        
        graph = get_graph_store()
        _upload_to_neo4j_from_pipeline(graph, result, filename)
        
        return {
            "success": True,
            "filename": filename,
            "doc_id": result.get("metadata", {}).get("doc_id"),
            "sections": len(result.get("sections", [])),
            "pipeline": "langgraph"
        }
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"그래프 업로드 실패: {str(e)}")


@app.get("/graph/documents")
def graph_list_documents():
    """Neo4j 문서 목록"""
    try:
        graph = get_graph_store()
        docs = graph.get_all_documents()
        return {"documents": docs, "count": len(docs)}
    except Exception as e:
        raise HTTPException(500, f"문서 목록 조회 실패: {str(e)}")


@app.get("/graph/document/{doc_id}")
def graph_get_document(doc_id: str):
    """특정 문서 상세"""
    try:
        graph = get_graph_store()
        doc = graph.get_document(doc_id)
        if not doc:
            raise HTTPException(404, f"문서를 찾을 수 없습니다: {doc_id}")
        return doc
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"문서 조회 실패: {str(e)}")


@app.delete("/graph/document/{doc_id}")
def graph_delete_document(doc_id: str):
    """Neo4j에서 문서 삭제"""
    try:
        graph = get_graph_store()
        result = graph.delete_document(doc_id)
        return {"success": True, "doc_id": doc_id, "result": result}
    except Exception as e:
        raise HTTPException(500, f"문서 삭제 실패: {str(e)}")


@app.get("/graph/document/{doc_id}/hierarchy")
def graph_get_hierarchy(doc_id: str):
    """문서 섹션 계층"""
    try:
        graph = get_graph_store()
        hierarchy = graph.get_section_hierarchy(doc_id)
        return {"doc_id": doc_id, "hierarchy": hierarchy}
    except Exception as e:
        raise HTTPException(500, f"계층 구조 조회 실패: {str(e)}")


@app.get("/graph/document/{doc_id}/references")
def graph_get_references(doc_id: str):
    """문서 참조 관계"""
    try:
        graph = get_graph_store()
        refs = graph.get_document_references(doc_id)
        if not refs:
            raise HTTPException(404, f"문서를 찾을 수 없습니다: {doc_id}")
        return refs
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"참조 조회 실패: {str(e)}")


@app.get("/graph/search/sections")
def graph_search_sections(keyword: str, doc_id: str = None):
    """섹션 검색"""
    try:
        graph = get_graph_store()
        results = graph.search_sections(keyword, doc_id)
        return {"keyword": keyword, "results": results, "count": len(results)}
    except Exception as e:
        raise HTTPException(500, f"검색 실패: {str(e)}")


@app.get("/graph/search/terms")
def graph_search_terms(term: str):
    """용어 검색 (간소화 버전: 섹션 검색으로 대체)"""
    try:
        graph = get_graph_store()
        results = graph.search_sections(term)
        return {"term": term, "results": results, "count": len(results)}
    except Exception as e:
        raise HTTPException(500, f"용어 검색 실패: {str(e)}")


# ═══════════════════════════════════════════════════════════════════════════
#  API 엔드포인트 - Question 추적
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/graph/questions")
def graph_list_questions(limit: int = 50, session_id: str = None):
    """질문 히스토리 조회"""
    try:
        graph = get_graph_store()
        questions = graph.get_question_history(session_id=session_id, limit=limit)
        return {"questions": questions, "count": len(questions)}
    except Exception as e:
        raise HTTPException(500, f"질문 조회 실패: {str(e)}")


@app.get("/graph/questions/{question_id}/sources")
def graph_get_question_sources(question_id: str):
    """질문이 참조한 섹션 조회"""
    try:
        graph = get_graph_store()
        result = graph.get_question_sources(question_id)
        if not result:
            raise HTTPException(404, f"질문을 찾을 수 없습니다: {question_id}")
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"소스 조회 실패: {str(e)}")


@app.get("/graph/stats/section-usage")
def graph_section_usage_stats(doc_id: str = None):
    """섹션 사용 통계 (간소화: Question 히스토리로 대체)"""
    try:
        graph = get_graph_store()
        # 간소화 버전: 전체 통계만 제공
        stats = graph.get_graph_stats()
        return {"stats": stats}
    except Exception as e:
        raise HTTPException(500, f"통계 조회 실패: {str(e)}")


# ═══════════════════════════════════════════════════════════════════════════
#  API 엔드포인트 - 에이전트 (NEW!)
# ═══════════════════════════════════════════════════════════════════════════

# 에이전트 모듈 임포트
try:
    from backend.agent import (
        init_agent_tools, 
        run_agent, 
        AGENT_TOOLS,
        LANGCHAIN_AVAILABLE,
        LANGGRAPH_AGENT_AVAILABLE,
        ZAI_AVAILABLE
    )
    AGENT_AVAILABLE = True
    print(" 에이전트 모듈 로드 완료")
except ImportError as e:
    AGENT_AVAILABLE = False
    LANGCHAIN_AVAILABLE = False
    LANGGRAPH_AGENT_AVAILABLE = False
    ZAI_AVAILABLE = False
    print(f" 에이전트 모듈 로드 실패: {e}")


class AgentRequest(BaseModel):
    """에이전트 요청"""
    message: str
    session_id: Optional[str] = None
    llm_model: str = "gpt-4o-mini"
    embedding_model: str = "multilingual-e5-small" # 추가
    n_results: int = DEFAULT_N_RESULTS #  추가
    use_langgraph: bool = True  # LangGraph 에이전트 사용 여부


@app.post("/agent/chat")
def agent_chat(request: AgentRequest):
    """
     에이전트 채팅 - LLM이 도구를 선택해서 실행
    
    일반 RAG와 다르게 에이전트가 상황에 맞는 도구를 선택합니다:
    - search_sop_documents: 문서 내용 검색
    - get_document_references: 문서 간 참조 관계
    - search_sections_by_keyword: 키워드로 섹션 검색
    - get_document_structure: 문서 구조/목차
    - list_all_documents: 전체 문서 목록
    """
    if not AGENT_AVAILABLE:
        raise HTTPException(500, "에이전트 모듈이 로드되지 않았습니다")
    
    session_id = request.session_id or str(uuid.uuid4())
    
    print(f"\n{'='*50}")
    print(f"[에이전트] 질문: {request.message}")
    print(f"  세션: {session_id}")
    print(f"  모드: {'LangGraph' if request.use_langgraph else 'Simple'}")
    print(f"  Orchestrator: gpt-4o-mini")
    print(f"  Worker: {request.llm_model}")

    try:
        # 도구 초기화 (처음 한 번만)
        init_agent_tools(vector_store, get_graph_store(), sql_store)
        
        # 통합된 멀티 에이전트 워크플로우 실행
        result = run_agent(
            query=request.message,
            session_id=session_id,
            model_name=request.llm_model,
            embedding_model=resolve_model_path(request.embedding_model)
        )
        
        reasoning = result.get("reasoning")
        answer = result.get("answer", "")

        # 본문(answer)이 비어있는데 reasoning만 있는 경우 (토큰 한도 초과 등으로 답변 생성 실패 시)
        if not answer and reasoning:
            print(" 본문이 직접적으로 수신되지 않아 사고 과정(Reasoning)을 답변으로 최우선 노출합니다.")
            result["answer"] = f"[AI 분석 리포트]\n\n{reasoning}"
            answer = result["answer"]
        
        if reasoning:
            print(f" 모델의 생각(Reasoning) 추출됨 ({len(reasoning)}자)")
            # 디버깅을 위해 첫 100자 정도 출력
            reasoning_preview = reasoning[:150].replace('\n', ' ')
            print(f"   [THINK] {reasoning_preview}...")
        
        print(f"   도구 호출: {len(result.get('tool_calls', []))}회")
        print(f"   답변 길이: {len(result.get('answer', ''))} 글자")
        print(f"{'='*50}\n")
        
        return {
            "session_id": session_id,
            "answer": result.get("answer", ""),
            "tool_calls": result.get("tool_calls", []),
            "success": result.get("success", False),
            "mode": "langgraph" if (request.use_langgraph and LANGGRAPH_AGENT_AVAILABLE) else "simple"
        }
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(500, f"에이전트 실행 실패: {str(e)}")


@app.get("/agent/status")
def agent_status():
    """에이전트 상태 확인"""
    return {
        "agent_available": AGENT_AVAILABLE,
        "langchain_available": LANGCHAIN_AVAILABLE if AGENT_AVAILABLE else False,
        "langgraph_agent_available": LANGGRAPH_AGENT_AVAILABLE if AGENT_AVAILABLE else False,
        "tools": [t.name for t in AGENT_TOOLS] if AGENT_AVAILABLE else [],
        "message": "에이전트 사용 가능" if AGENT_AVAILABLE else "에이전트 모듈 로드 실패"
    }


@app.get("/agent/tools")
def agent_tools():
    """에이전트 도구 목록"""
    if not AGENT_AVAILABLE:
        raise HTTPException(500, "에이전트 모듈이 로드되지 않았습니다")

    tools_info = []
    for tool in AGENT_TOOLS:
        tools_info.append({
            "name": tool.name,
            "description": tool.description
        })

    return {"tools": tools_info, "count": len(tools_info)}


#  테스트용 간단한 에코 엔드포인트
class SimpleRequest(BaseModel):
    message: str

@app.post("/test/echo")
def test_echo(request: SimpleRequest):
    """테스트용 간단한 에코 API"""
    return {
        "session_id": str(uuid.uuid4()),
        "answer": f"테스트 응답: {request.message}",
        "success": True
    }


# ═══════════════════════════════════════════════════════════════════════════
# API 엔드포인트 - LLM as a Judge 평가
# ═══════════════════════════════════════════════════════════════════════════

class EvaluationRequest(BaseModel):
    """평가 요청 모델"""
    question: str
    answer: str
    context: Optional[str] = ""
    metrics: Optional[List[str]] = None  # ["faithfulness", "groundness", "relevancy", "correctness"]
    reference_answer: Optional[str] = None

@app.post("/evaluate")
def evaluate_answer(request: EvaluationRequest):
    """
    🔍 LLM as a Judge - 답변 평가 (RDB 검증 포함)

    평가 메트릭:
    - faithfulness: 컨텍스트 충실성 (환각 방지)
    - groundness: 근거 명확성
    - relevancy: 질문 관련성
    - correctness: 정확성과 완전성

    **무조건 RDB에서 실제 문서를 조회하여 인용 정확성 검증**
    """
    try:
        from backend.evaluation import AgentEvaluator

        # RDB 검증을 위해 sql_store 필수 전달
        evaluator = AgentEvaluator(
            judge_model="gpt-4o-mini",
            sql_store=sql_store
        )

        # 평가 실행
        results = evaluator.evaluate_single(
            question=request.question,
            answer=request.answer,
            context=request.context,
            metrics=request.metrics,
            reference_answer=request.reference_answer
        )

        return {
            "success": True,
            "evaluation": results,
            "average_score": results.get("average_score", 0)
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(500, f"평가 실행 실패: {str(e)}")


# ═══════════════════════════════════════════════════════════════════════════
# 서버 실행
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("[시스템] 초기화 중...")
    sql_store.init_db()
    
    # Neo4j 연결 확인 (성공 로그는 connect 내부에서 출력됨)
    try:
        get_graph_store()
    except Exception as e:
        print(f" Neo4j 초기 연결 실패: {e}")

    # Weaviate 연결 확인 (성공 로그는 get_client 내부에서 출력됨)
    try:
        wv_client = vector_store.get_client()
        if not wv_client.is_connected():
            print(" Weaviate v4 연결 상태 확인 실패")
    except Exception as e:
        print(f" Weaviate v4 연결 체크 중 오류: {e}")

    
    import uvicorn
    
    print("\n" + "=" * 60)
    print(" RAG Chatbot API v14.0 + OpenAI Agent")
    print("=" * 60)
    print(f" LLM 백엔드: OpenAI (GPT-4o-mini)")
    print(f" 에이전트: {' 활성화' if AGENT_AVAILABLE else ' 비활성화'}")
    
    if AGENT_AVAILABLE:
        print(f"   - LangChain: {'' if LANGCHAIN_AVAILABLE else ''}")
    print("Docs: http://localhost:8000/docs")
    print("=" * 60)
    print("주요 기능:")
    print("  - LangGraph 문서 파이프라인")
    print("  -  ReAct 에이전트 (/agent/chat)")
    print("  - Weaviate(v4) + Neo4j + PostgreSQL")
    print("  - LangSmith 추적 지원")
    print("=" * 60)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()