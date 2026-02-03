"""
RAG 챗봇 API v11.0 + Agent (Z.AI)

🔥 v11.0 변경사항:
- LLM 백엔드 변경: Ollama → Z.AI GLM-4.7-Flash
- 에이전트 도구 성능 강화
- LangSmith 추적 지원 및 최적화
- 되묻기 로직 제거 및 검색 결과 직접 출력
"""

# 🔥 .env 파일 자동 로드 (다른 import보다 먼저!)
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import torch
import time
import uuid

from backend.sql_store import SQLStore
sql_store = SQLStore()
sql_store.init_db()

# RAG 모듈 - 레거시 (폴백용)
from backend import (
    load_document,
    get_supported_extensions,
    create_chunks,
    create_chunks_from_blocks,
    get_available_methods,
    CHUNK_METHODS,
)
from backend import vector_store
from backend.prompt import build_rag_prompt, build_chunk_prompt
from backend.llm import (
    get_llm_response,
    ZaiLLM,
    OllamaLLM,
    analyze_search_results,
    HUGGINGFACE_MODELS,
)

# 🔥 LangGraph 파이프라인 (v9.2)
try:
    from backend.document_pipeline import process_document, state_to_chunks, Chunk
    LANGGRAPH_AVAILABLE = True
    print("✅ LangGraph 파이프라인 사용 가능")
except ImportError as e:
    LANGGRAPH_AVAILABLE = False
    print(f"⚠️ LangGraph 사용 불가, 레거시 모드: {e}")


app = FastAPI(title="RAG Chatbot API", version="9.2.0")

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
DEFAULT_N_RESULTS = 7  # 🔥 5 -> 7 상향
DEFAULT_SIMILARITY_THRESHOLD = 0.30  # 🔥 0.35 -> 0.30 (더 많은 맥락 확보)
USE_LANGGRAPH = True  # 🔥 LangGraph 파이프라인 사용 여부

PRESET_MODELS = {
    "ko-sbert": "snunlp/KR-SBERT-V40K-klueNLI-augSTS",
    "ko-simcse": "BM-K/KoSimCSE-roberta",
    "multilingual-minilm": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "multilingual-e5-large": "intfloat/multilingual-e5-large",
    "multilingual-e5-small": "intfloat/multilingual-e5-small",
    "bge-m3": "BAAI/bge-m3",
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "mpnet": "sentence-transformers/all-mpnet-base-v2",
    "qwen3-0.6b": "Qwen/Qwen3-Embedding-0.6B",
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
    llm_model: str = "qwen2.5:3b"
    llm_backend: str = "ollama"
    filter_doc: Optional[str] = None
    similarity_threshold: Optional[float] = None


class AskRequest(BaseModel):
    query: str
    collection: str = "documents"
    n_results: int = DEFAULT_N_RESULTS
    embedding_model: str = "multilingual-e5-small"
    llm_model: str = "glm-4.7-flash"
    llm_backend: str = "zai"  # 🔥 기본값 zai로 변경
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
    delete_from_neo4j: bool = True  # 🔥 Neo4j에서도 삭제


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
        
        # 🔥 v9.2: 개선된 출처 표시
        sop_id = meta.get("sop_id", "")
        section_path = meta.get("section_path", "")
        page = meta.get("page", "")
        article_num = meta.get("article_num", "")
        
        # 출처 헤더 구성
        source_parts = []
        if sop_id:
            source_parts.append(f"[{sop_id}]")
        if section_path:
            source_parts.append(f"> {section_path}")
        if page:
            source_parts.append(f"(p.{page})")
        if similarity:
            source_parts.append(f"관련도: {similarity:.0%}")
        
        source_header = " ".join(source_parts) if source_parts else f"[문서 {i}]"
        
        context_parts.append(f"📄 {source_header}\n{text}")
    
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
            "ChromaDB + Neo4j 동기화 삭제"
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
# 🔥 API 엔드포인트 - 업로드 (LangGraph v9.2)
# ═══════════════════════════════════════════════════════════════════════════

@app.post("/rag/upload")
async def upload_document(
    file: UploadFile = File(...),
    collection: str = Form("documents"),
    chunk_size: int = Form(DEFAULT_CHUNK_SIZE),
    chunk_method: str = Form(DEFAULT_CHUNK_METHOD),
    model: str = Form("multilingual-e5-small"),
    overlap: int = Form(DEFAULT_OVERLAP),
    use_langgraph: bool = Form(True),  # 🔥 LangGraph 사용 여부
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
        print(f"📄 문서 업로드: {filename}")
        print(f"{'='*70}")
        
        # 🔥 LangGraph 파이프라인 vs 레거시 선택
        if use_langgraph and LANGGRAPH_AVAILABLE and USE_LANGGRAPH:
            # === LangGraph 파이프라인 (v9.2) ===
            print(f"   🔥 LangGraph 파이프라인 사용")
            
            result = process_document(
                filename=filename,
                content=content,
                chunk_size=chunk_size,
                chunk_overlap=overlap,
                debug=True
            )
            
            if not result.get("success"):
                errors = result.get("errors", ["알 수 없는 오류"])
                raise HTTPException(400, f"문서 처리 실패: {errors}")
            
            chunks = state_to_chunks(result)
            
            # 메타데이터 보강
            metadata_base = result.get("metadata", {})
            sop_id = metadata_base.get("doc_id") or metadata_base.get("sop_id")
            
            # 🔥 ID가 없으면 파일명에서 끝자리 숫자로라도 유추 시도
            if not sop_id:
                import re
                id_match = re.search(r'([A-Z0-9]+-[A-Z0-9]+-\d+)', filename)
                if id_match:
                    sop_id = id_match.group(1)
                else:
                    sop_id = filename.split('.')[0] # 최후의 수단: 파일명
            
            # 제목 설정: 원본 파일명 유지 (사용자 요청)
            doc_title = filename 
            extracted_title = metadata_base.get("title")
            if extracted_title and extracted_title not in filename:
                doc_title = f"{filename} ({extracted_title})"
            
            print(f"   DOC ID: {sop_id}")
            print(f"   제목: {doc_title}")
            print(f"   품질 점수: {result.get('quality_score', 0):.0%}")
            print(f"   변환 방법: {result.get('conversion_method')}")
            print(f"   총 청크: {len(chunks)}")
            
            pipeline_version = "langgraph-v9.2"
            quality_score = result.get("quality_score", 0)
            conversion_method = result.get("conversion_method", "unknown")
            
        else:
            # === 레거시 파이프라인 ===
            print(f"   📦 레거시 파이프라인 사용")
            
            parsed_doc = load_document(filename, content)
            sop_id = parsed_doc.metadata.get("sop_id")
            doc_title = parsed_doc.metadata.get("title", filename)
            
            print(f"   SOP ID: {sop_id}")
            print(f"   제목: {doc_title}")
            print(f"   총 블록 수: {len(parsed_doc.blocks)}")
            
            if chunk_method == "article" and parsed_doc.blocks:
                chunks = create_chunks_from_blocks(
                    parsed_doc,
                    chunk_size=chunk_size,
                    overlap=overlap,
                    method="recursive",
                    exclude_intro=True,
                )
            else:
                chunks = create_chunks(
                    parsed_doc.text,
                    chunk_size=chunk_size,
                    overlap=overlap,
                    method=chunk_method
                )
                for chunk in chunks:
                    chunk.metadata.update({
                        "doc_name": filename,
                        "doc_title": doc_title,
                        "sop_id": sop_id,
                    })
            
            # 빈 청크 체크
            if not chunks:
                chunks = create_chunks_from_blocks(
                    parsed_doc,
                    chunk_size=chunk_size,
                    overlap=overlap,
                    method="recursive",
                    exclude_intro=False,
                )
            
            pipeline_version = "legacy-v6.3"
            quality_score = None
            conversion_method = "legacy"
            metadata_base = parsed_doc.metadata
        
        if not chunks:
            raise HTTPException(400, "문서에서 텍스트를 추출할 수 없습니다.")
        
        # === ChromaDB 저장 ===
        model_path = resolve_model_path(model)
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
        print(f"   ✅ ChromaDB 저장 완료: {len(chunks)} 청크")
        
        # === PostgreSQL 저장 ===
        try:
            # 원본 마크다운 결정 (LangGraph 결과 우선, 없으면 청크 합산)
            full_markdown = ""
            if use_langgraph and 'result' in locals() and result.get("markdown"):
                full_markdown = result.get("markdown")
            else:
                full_markdown = "\n\n".join([c.text for c in chunks])

            sql_store.save_document(
                sop_id=sop_id,
                title=doc_title or filename,
                markdown_content=full_markdown,
                pdf_binary=content if filename.lower().endswith(".pdf") else None,
                doc_metadata={
                    "doc_id": metadata_base.get("doc_id"),
                    "version": metadata_base.get("version"),
                    "effective_date": metadata_base.get("effective_date"),
                    "owning_dept": metadata_base.get("owning_dept"),
                    "filename": filename
                }
            )
        except Exception as sql_err:
            print(f"   ⚠️ PostgreSQL 상세 저장 실패: {sql_err}")
            # 폴백: 기존 유저 코드 방식 (필요 시)
            # save_chunks_to_db(sop_id, filename, chunks)
        
        # === Neo4j 그래프 저장 ===
        graph_uploaded = False
        graph_sections = 0
        
        try:
            from backend.graph_store import Neo4jGraphStore
            
            graph = get_graph_store()
            if graph.test_connection():
                # LangGraph 결과에서 그래프 생성
                if use_langgraph and LANGGRAPH_AVAILABLE:
                    # 직접 섹션 데이터로 그래프 생성
                    _upload_to_neo4j_from_pipeline(graph, result, filename)
                else:
                    # 레거시: ParsedDocument에서 생성
                    from backend.graph_store import document_to_graph
                    document_to_graph(graph, parsed_doc, sop_id)
                
                graph_uploaded = True
                stats = graph.get_graph_stats()
                graph_sections = stats.get("sections", 0)
                print(f"   ✅ Neo4j 그래프 업로드 완료")
        except Exception as graph_error:
            print(f"   ⚠️ Neo4j 그래프 업로드 실패 (무시됨): {graph_error}")
        
        elapsed = round(time.time() - start_time, 2)
        
        return {
            "success": True,
            "filename": filename,
            "sop_id": sop_id,
            "doc_title": doc_title,
            "chunks": len(chunks),
            "chunk_method": chunk_method,
            "pipeline_version": pipeline_version,
            "quality_score": quality_score,
            "conversion_method": conversion_method,
            "graph_uploaded": graph_uploaded,
            "elapsed_seconds": elapsed,
            "sample_metadata": metadatas[0] if metadatas else {},
        }
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(500, f"업로드 실패: {str(e)}")


def _upload_to_neo4j_from_pipeline(graph, result: dict, filename: str):
    """LangGraph 파이프라인 결과를 Neo4j에 업로드 (V22.0 대응)"""
    metadata = result.get("metadata", {})
    doc_id = metadata.get("doc_id") or "UNKNOWN"
    title = metadata.get("title") or filename
    version = metadata.get("version") or "1.0"
    effective_date = metadata.get("effective_date")
    owning_dept = metadata.get("owning_dept")
    
    # 1. Document 노드 생성
    graph.create_document(
        doc_id=doc_id,
        title=title,
        version=version,
        effective_date=effective_date,
        owning_dept=owning_dept,
        metadata=metadata
    )
    
    # 2. DocumentType 처리 (코드 기반)
    doc_type_code = "SOP" # 기본값
    if "SOP" in doc_id: doc_type_code = "SOP"
    elif "WI" in doc_id: doc_type_code = "WI"
    
    graph.create_document_type(doc_type_code, "표준작업절차서" if doc_type_code == "SOP" else "작업지침서", doc_type_code)
    graph.link_doc_to_type(doc_id, doc_type_code)
    
    # 3. Section 노드 생성 및 관계 설정
    sections = result.get("sections", [])
    
    for sec in sections:
        headers = sec.get("headers", {})
        content = sec.get("content", "")
        page = sec.get("page", 1)
        parent_name = sec.get("parent")
        clause_meta = sec.get("clause_meta", {})
        
        # clause_level 및 section_id 유추
        clause_level = 0
        current_title = ""
        for level in range(6, 0, -1):
            if headers.get(f"H{level}"):
                clause_level = level
                current_title = headers[f"H{level}"]
                break
        
        clause_id = None
        import re
        num_match = re.match(r'^(\d+(?:\.\d+)*)', current_title)
        if num_match:
            clause_id = num_match.group(1)
        
        if not clause_id: continue
        
        section_id = f"{doc_id}:{clause_id}"
        main_section = clause_id.split('.')[0] if '.' in clause_id else clause_id
        
        # Section 노드 생성 (상세 메타데이터 포함)
        graph.create_section(
            doc_id=doc_id,
            section_id=section_id,
            title=current_title,
            content=content,
            clause_level=clause_level,
            main_section=main_section,
            llm_meta=clause_meta,
            page=page
        )
        
        # 4. 계층 관계 (Parent-Child)
        if parent_name:
            # 부모 ID 유추 (단순화: 같은 문서 내에서 점 하나 뺀 패턴)
            if '.' in clause_id:
                parent_clause_id = '.'.join(clause_id.split('.')[:-1])
                parent_section_id = f"{doc_id}:{parent_clause_id}"
                graph.create_section_hierarchy(parent_section_id, section_id)
        
        # 5. Concept 연동 (intent_scope 활용)
        intent_scope = clause_meta.get("intent_scope")
        if intent_scope:
            graph.create_concept(intent_scope, intent_scope, intent_scope)
            graph.link_section_to_concept(section_id, intent_scope)
            
        # 6. 타 문서 언급 (MENTIONS) 추적
        mentions = re.findall(r'((?:EQ-)?SOP[-_]?\d{4,5})', content, re.IGNORECASE)
        for m in set(mentions):
            m_id = m.upper().replace('_', '-')
            if not m_id.startswith('EQ-'): m_id = 'EQ-' + m_id
            if m_id != doc_id:
                graph.link_section_to_mention_doc(section_id, m_id)


# ═══════════════════════════════════════════════════════════════════════════
# API 엔드포인트 - 검색
# ═══════════════════════════════════════════════════════════════════════════

@app.post("/rag/search")
def search_documents(request: SearchRequest):
    """벡터 검색"""
    model_path = resolve_model_path(request.model)
    threshold = request.similarity_threshold or DEFAULT_SIMILARITY_THRESHOLD
    
    results = vector_store.search(
        query=request.query,
        collection_name=request.collection,
        model_name=model_path,
        n_results=request.n_results,
        filter_doc=request.filter_doc,
        similarity_threshold=threshold,
    )
    
    return {
        "query": request.query,
        "results": results,
        "count": len(results),
        "threshold": threshold,
    }


@app.post("/rag/search/advanced")
def search_advanced(request: SearchRequest):
    """고급 검색 (품질 메트릭 포함)"""
    model_path = resolve_model_path(request.model)
    threshold = request.similarity_threshold or DEFAULT_SIMILARITY_THRESHOLD
    
    response = vector_store.search_advanced(
        query=request.query,
        collection_name=request.collection,
        model_name=model_path,
        n_results=request.n_results,
        filter_doc=request.filter_doc,
        similarity_threshold=threshold,
    )
    
    return response


# ═══════════════════════════════════════════════════════════════════════════
# API 엔드포인트 - 챗봇
# ═══════════════════════════════════════════════════════════════════════════

@app.post("/chat")
def chat(request: ChatRequest):
    """대화형 RAG 챗봇 (v10.1 - 되묻기/추적 제거, 소스 형식 수정)"""
    session_id = request.session_id or str(uuid.uuid4())
    
    if session_id not in chat_histories:
        chat_histories[session_id] = []
    
    # 1. 벡터 검색
    model_path = resolve_model_path(request.embedding_model)
    threshold = request.similarity_threshold or DEFAULT_SIMILARITY_THRESHOLD
    
    results, context = vector_store.search_with_context(
        query=request.message,
        collection_name=request.collection,
        model_name=model_path,
        n_results=request.n_results,
        filter_doc=request.filter_doc,
        similarity_threshold=threshold,
    )
    
    # 🔥 [조정] 컨텍스트가 극단적으로 짧을 때만(예: 400자 미만) 최소한의 보충 수행
    if len(context) < 400 and len(results) < 10:
        print(f"⚠️ 컨텍스트가 너무 짧음({len(context)}자). 최소 보충 시도...")
        extra_results, extra_context = vector_store.search_with_context(
            query=request.message,
            collection_name=request.collection,
            model_name=model_path,
            n_results=10,  # 15에서 10으로 하향
            filter_doc=request.filter_doc,
            similarity_threshold=threshold * 0.7, # 임계값 더 완화하여 주변 맥락 확보
        )
        if len(extra_context) > len(context):
            results, context = extra_results, extra_context
            print(f"✅ 컨텍스트 보충 완료 ({len(context)}자, {len(results)}개)")
    
    # 🔥 디버그 로그
    print(f"\n{'='*50}")
    print(f"🔍 질문: {request.message}")
    print(f"📊 검색 결과: {len(results)}개")
    if results:
        for i, r in enumerate(results[:3]):
            sim = r.get('similarity', 0)
            meta = r.get('metadata', {})
            sop = meta.get('sop_id', '?')
            path = meta.get('section_path', '')[:40] if meta.get('section_path') else ''
            print(f"   [{i+1}] 유사도: {sim:.2f} | {sop} | {path}...")
    print(f"📝 컨텍스트 길이: {len(context)} 글자")
    
    # 2. LLM 답변 생성 (되묻기 로직 제거!)
    prompt = build_rag_prompt(request.message, context)
    print(f"🤖 LLM 호출 시도:")
    print(f"   - Backend: {request.llm_backend}")
    print(f"   - Model: {request.llm_model}")
    print(f"   - Prompt Prefix: {prompt[:100]}...")
    
    answer = get_llm_response(
        prompt=prompt,
        llm_model=request.llm_model,
        llm_backend=request.llm_backend,
        max_tokens=2048 # 🔥 Z.AI 분석과 한국어 답변을 위해 2048 권장
    )
    
    print(f"💬 LLM 결과:")
    print(f"   - 답변: {answer[:50]}..." if answer else "   - 답변: (EMPTY)")
    print(f"   - 길이: {len(answer)} 글자")
    print(f"{'='*50}\n")
    
    # 3. 히스토리 저장
    chat_histories[session_id].append({"role": "user", "content": request.message})
    chat_histories[session_id].append({"role": "assistant", "content": answer})
    
    # 4. 소스 정보 (프론트엔드 형식에 맞춤!)
    sources = []
    for r in results:
        meta = r.get("metadata", {})
        sources.append({
            "text": r.get("text", ""),
            "similarity": r.get("similarity", 0),
            "metadata": meta,
            "metadata_display": {
                "doc_name": meta.get("doc_name", "문서"),
                "doc_title": meta.get("doc_title", ""),
                "sop_id": meta.get("sop_id", ""),
                "version": meta.get("version", ""),
                "section": meta.get("section", ""),
                "section_path": meta.get("section_path", ""),
                "section_path_readable": meta.get("section_path_readable", ""),
                "title": meta.get("title", ""),
                "page": meta.get("page", ""),
            }
        })
    
    return {
        "session_id": session_id,
        "answer": answer,
        "sources": sources,
        "needs_clarification": False
    }


@app.post("/rag/ask")
def ask_with_rag(request: AskRequest):
    """단일 질문 RAG (Question 추적 포함)"""
    model_path = resolve_model_path(request.embedding_model)
    threshold = request.similarity_threshold or DEFAULT_SIMILARITY_THRESHOLD
    
    # 검색
    results, context = vector_store.search_with_context(
        query=request.query,
        collection_name=request.collection,
        model_name=model_path,
        n_results=request.n_results,
        filter_doc=request.filter_doc,
        similarity_threshold=threshold,
    )
    
    if not results:
        return {
            "query": request.query,
            "answer": "관련 문서를 찾을 수 없습니다.",
            "sources": [],
            "context_used": ""
        }
    
    # 🔥 개선된 컨텍스트 포맷팅
    formatted_context = format_context(results)
    prompt = build_rag_prompt(request.query, formatted_context, request.language)
    
    answer = get_llm_response(
        prompt=prompt,
        llm_model=request.llm_model,
        llm_backend=request.llm_backend,
        max_tokens=request.max_tokens
    )
    
    # 🔥 Question 추적 (Neo4j)
    question_id = None
    try:
        from backend.graph_store import track_rag_question
        graph = get_graph_store()
        if graph.test_connection():
            question_id = track_rag_question(
                graph_store=graph,
                question_text=request.query,
                search_results=results,
                answer=answer,
                embedding_model=request.embedding_model,
                llm_model=request.llm_model
            )
    except Exception as e:
        print(f"⚠️ Question 추적 실패: {e}")
    
    sources = []
    if request.include_sources:
        sources = [
            {
                "doc_name": r.get("metadata", {}).get("doc_name"),
                "sop_id": r.get("metadata", {}).get("sop_id"),
                "section": r.get("metadata", {}).get("section"),
                "section_path": r.get("metadata", {}).get("section_path"),
                "page": r.get("metadata", {}).get("page"),
                "similarity": r.get("similarity"),
                "confidence": r.get("confidence"),
            }
            for r in results
        ]
    
    return {
        "query": request.query,
        "answer": answer,
        "sources": sources,
        "question_id": question_id,
        "context_used": formatted_context[:500] + "..." if len(formatted_context) > 500 else formatted_context
    }


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
    🔥 문서 삭제 (ChromaDB + Neo4j 동시 삭제)
    """
    result = {"chromadb": None, "neo4j": None}
    
    # 1. ChromaDB 삭제
    chroma_result = vector_store.delete_by_doc_name(
        doc_name=request.doc_name,
        collection_name=request.collection
    )
    result["chromadb"] = chroma_result
    
    # 2. Neo4j 삭제 (옵션)
    if request.delete_from_neo4j:
        try:
            graph = get_graph_store()
            if graph.test_connection():
                # doc_name에서 sop_id 추출 시도
                import re
                sop_match = re.search(r'(EQ-SOP-\d+)', request.doc_name, re.IGNORECASE)
                if sop_match:
                    sop_id = sop_match.group(1).upper()
                    neo4j_result = graph.delete_document(sop_id)
                    result["neo4j"] = {"success": True, "sop_id": sop_id, "deleted": neo4j_result}
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
        
        if use_langgraph and LANGGRAPH_AVAILABLE:
            result = process_document(filename, content, debug=True)
            if not result.get("success"):
                raise HTTPException(400, f"처리 실패: {result.get('errors')}")
            
            graph = get_graph_store()
            _upload_to_neo4j_from_pipeline(graph, result, filename)
            
            return {
                "success": True,
                "filename": filename,
                "sop_id": result.get("metadata", {}).get("sop_id"),
                "sections": len(result.get("sections", [])),
                "pipeline": "langgraph"
            }
        else:
            from backend.graph_store import document_to_graph
            
            parsed_doc = load_document(filename, content)
            sop_id = parsed_doc.metadata.get("sop_id")
            
            graph = get_graph_store()
            document_to_graph(graph, parsed_doc, sop_id)
            
            return {
                "success": True,
                "filename": filename,
                "sop_id": sop_id,
                "blocks": len(parsed_doc.blocks),
                "pipeline": "legacy"
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


@app.get("/graph/document/{sop_id}")
def graph_get_document(sop_id: str):
    """특정 문서 상세"""
    try:
        graph = get_graph_store()
        doc = graph.get_document(sop_id)
        if not doc:
            raise HTTPException(404, f"문서를 찾을 수 없습니다: {sop_id}")
        return doc
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"문서 조회 실패: {str(e)}")


@app.delete("/graph/document/{sop_id}")
def graph_delete_document(sop_id: str):
    """Neo4j에서 문서 삭제"""
    try:
        graph = get_graph_store()
        result = graph.delete_document(sop_id)
        return {"success": True, "sop_id": sop_id, "result": result}
    except Exception as e:
        raise HTTPException(500, f"문서 삭제 실패: {str(e)}")


@app.get("/graph/document/{sop_id}/hierarchy")
def graph_get_hierarchy(sop_id: str):
    """문서 섹션 계층"""
    try:
        graph = get_graph_store()
        hierarchy = graph.get_section_hierarchy(sop_id)
        return {"sop_id": sop_id, "hierarchy": hierarchy}
    except Exception as e:
        raise HTTPException(500, f"계층 구조 조회 실패: {str(e)}")


@app.get("/graph/document/{sop_id}/references")
def graph_get_references(sop_id: str):
    """문서 참조 관계"""
    try:
        graph = get_graph_store()
        refs = graph.get_document_references(sop_id)
        if not refs:
            raise HTTPException(404, f"문서를 찾을 수 없습니다: {sop_id}")
        return refs
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"참조 조회 실패: {str(e)}")


@app.get("/graph/search/sections")
def graph_search_sections(keyword: str, sop_id: str = None):
    """섹션 검색"""
    try:
        graph = get_graph_store()
        results = graph.search_sections(keyword, sop_id)
        return {"keyword": keyword, "results": results, "count": len(results)}
    except Exception as e:
        raise HTTPException(500, f"검색 실패: {str(e)}")


@app.get("/graph/search/terms")
def graph_search_terms(term: str):
    """용어 검색"""
    try:
        graph = get_graph_store()
        results = graph.search_by_term(term)
        return {"term": term, "results": results, "count": len(results)}
    except Exception as e:
        raise HTTPException(500, f"용어 검색 실패: {str(e)}")


# ═══════════════════════════════════════════════════════════════════════════
# 🔥 API 엔드포인트 - Question 추적
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
def graph_section_usage_stats(sop_id: str = None):
    """섹션 사용 통계"""
    try:
        graph = get_graph_store()
        stats = graph.get_section_usage_stats(sop_id)
        return {"stats": stats, "count": len(stats)}
    except Exception as e:
        raise HTTPException(500, f"통계 조회 실패: {str(e)}")


# ═══════════════════════════════════════════════════════════════════════════
# 🤖 API 엔드포인트 - 에이전트 (NEW!)
# ═══════════════════════════════════════════════════════════════════════════

# 에이전트 모듈 임포트
try:
    from backend.agent import (
        init_agent_tools, 
        run_agent, 
        create_agent,
        AGENT_TOOLS,
        LANGCHAIN_AVAILABLE,
        LANGGRAPH_AGENT_AVAILABLE,
        ZAI_AVAILABLE
    )
    AGENT_AVAILABLE = True
    print("✅ 에이전트 모듈 로드 완료")
except ImportError as e:
    AGENT_AVAILABLE = False
    LANGCHAIN_AVAILABLE = False
    LANGGRAPH_AGENT_AVAILABLE = False
    ZAI_AVAILABLE = False
    print(f"⚠️ 에이전트 모듈 로드 실패: {e}")


class AgentRequest(BaseModel):
    """에이전트 요청"""
    message: str
    session_id: Optional[str] = None
    llm_model: str = "glm-4.7-flash"
    embedding_model: str = "multilingual-e5-small" # 추가
    n_results: int = DEFAULT_N_RESULTS # 🔥 추가
    use_langgraph: bool = True  # LangGraph 에이전트 사용 여부


@app.post("/agent/chat")
def agent_chat(request: AgentRequest):
    """
    🤖 에이전트 채팅 - LLM이 도구를 선택해서 실행
    
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
    print(f"🤖 에이전트 질문: {request.message}")
    print(f"   세션: {session_id}")
    print(f"   모드: {'LangGraph' if request.use_langgraph else 'Simple'}")
    
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
            print("⚠️ 본문이 직접적으로 수신되지 않아 사고 과정(Reasoning)을 답변으로 최우선 노출합니다.")
            result["answer"] = f"[AI 분석 리포트]\n\n{reasoning}"
            answer = result["answer"]
        
        if reasoning:
            print(f"🧠 모델의 생각(Reasoning) 추출됨 ({len(reasoning)}자)")
            # 디버깅을 위해 첫 100자 정도 출력
            print(f"   [THINK] {reasoning[:150].replace('\n', ' ')}...")
        
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


# 🔥 테스트용 간단한 에코 엔드포인트
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
# 서버 실행
# ═══════════════════════════════════════════════════════════════════════════

def main():
    sql_store.init_db()
    print("🚀 RAG 시스템 시작")
    
    import uvicorn
    
    print("\n" + "=" * 60)
    print("🤖 RAG Chatbot API v11.0 + Z.AI Agent")
    print("=" * 60)
    print(f"🔥 LLM 백엔드: {'✅ Z.AI (GLM-4.7-Flash)' if ZaiLLM.is_available() else '❌ ZAI_API_KEY 설정 필요'}")
    print(f"🤖 에이전트: {'✅ 활성화' if AGENT_AVAILABLE else '❌ 비활성화'}")
    
    if AGENT_AVAILABLE:
        print(f"   - LangChain: {'✅' if LANGCHAIN_AVAILABLE else '❌'}")
    print("Docs: http://localhost:8000/docs")
    print("=" * 60)
    print("주요 기능:")
    print("  - LangGraph 문서 파이프라인")
    print("  - 🤖 ReAct 에이전트 (/agent/chat)")
    print("  - ChromaDB + Neo4j + PostgreSQL")
    print("  - LangSmith 추적 지원")
    print("=" * 60)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()