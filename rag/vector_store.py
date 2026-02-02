"""
ChromaDB 벡터 스토어 - v6.0
- search() 함수에 similarity_threshold 추가
- confidence 레벨 계산 정확히 구현
- 검색 품질 지표 포함
"""

import chromadb
from chromadb.config import Settings
import numpy as np
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import hashlib
import torch
from transformers import AutoTokenizer, AutoModel
from dataclasses import dataclass


# ═══════════════════════════════════════════════════════════════════════════
# 설정 상수
# ═══════════════════════════════════════════════════════════════════════════

DEFAULT_COLLECTION = "documents"
CHROMA_PATH = "./chroma_db"

# 검색 품질 설정
DEFAULT_SIMILARITY_THRESHOLD = 0.35
HIGH_CONFIDENCE_THRESHOLD = 0.65
MEDIUM_CONFIDENCE_THRESHOLD = 0.45
MIN_RESULTS_BEFORE_FILTER = 1

# 임베딩 모델 필터링 기준
MAX_EMBEDDING_DIM = 1024
MAX_MEMORY_MB = 1300

# 전역 캐시
_client: Optional[chromadb.PersistentClient] = None
_embed_models: Dict = {}
_device: Optional[str] = None


# ═══════════════════════════════════════════════════════════════════════════
# 임베딩 모델 스펙
# ═══════════════════════════════════════════════════════════════════════════

EMBEDDING_MODEL_SPECS = {
    "snunlp/KR-SBERT-V40K-klueNLI-augSTS": {
        "name": "ko-sbert", "dim": 768, "memory_mb": 440, "lang": "ko",
    },
    "BM-K/KoSimCSE-roberta": {
        "name": "ko-simcse", "dim": 768, "memory_mb": 440, "lang": "ko",
    },
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": {
        "name": "multilingual-minilm", "dim": 384, "memory_mb": 470, "lang": "multi",
    },
    "intfloat/multilingual-e5-large": {
        "name": "multilingual-e5-large", "dim": 1024, "memory_mb": 1200, "lang": "multi",
    },
    "intfloat/multilingual-e5-small": {
        "name": "multilingual-e5-small", "dim": 384, "memory_mb": 120, "lang": "multi",
    },
    "BAAI/bge-m3": {
        "name": "bge-m3", "dim": 1024, "memory_mb": 1300, "lang": "multi",
    },
    "sentence-transformers/all-MiniLM-L6-v2": {
        "name": "minilm", "dim": 384, "memory_mb": 90, "lang": "en",
    },
    "sentence-transformers/all-mpnet-base-v2": {
        "name": "mpnet", "dim": 768, "memory_mb": 420, "lang": "en",
    },
    "Qwen/Qwen3-Embedding-0.6B": {
        "name": "qwen3-0.6b", "dim": 1024, "memory_mb": 600, "lang": "multi",
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# 검색 결과 데이터 클래스
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class SearchResult:
    """검색 결과 단일 항목"""
    text: str
    similarity: float
    metadata: Dict
    id: str
    confidence: str

    def to_dict(self) -> Dict:
        return {
            "text": self.text,
            "similarity": self.similarity,
            "metadata": self.metadata,
            "id": self.id,
            "confidence": self.confidence,
        }


@dataclass
class SearchResponse:
    """검색 응답 전체"""
    results: List[SearchResult]
    query: str
    total_found: int
    filtered_count: int
    quality_summary: Dict

    def to_dict(self) -> Dict:
        return {
            "results": [r.to_dict() for r in self.results],
            "query": self.query,
            "total_found": self.total_found,
            "filtered_count": self.filtered_count,
            "quality_summary": self.quality_summary,
        }


# ═══════════════════════════════════════════════════════════════════════════
# 유틸리티 함수
# ═══════════════════════════════════════════════════════════════════════════

def get_device() -> str:
    global _device
    if _device is None:
        _device = "cuda" if torch.cuda.is_available() else "cpu"
    return _device


def get_client() -> chromadb.PersistentClient:
    global _client
    if _client is None:
        Path(CHROMA_PATH).mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(path=CHROMA_PATH)
    return _client


def get_collection_name_for_model(base_name: str, model_name: str) -> str:
    """모델별 컬렉션 이름 생성"""
    model_hash = hashlib.md5(model_name.encode()).hexdigest()[:8]
    return f"{base_name}__{model_hash}"


def calculate_confidence(similarity: float) -> str:
    """유사도 기반 신뢰도 계산"""
    if similarity >= HIGH_CONFIDENCE_THRESHOLD:
        return "high"
    elif similarity >= MEDIUM_CONFIDENCE_THRESHOLD:
        return "medium"
    return "low"


def get_embedding_model_info() -> Dict:
    """임베딩 모델 전체 정보"""
    all_models = []
    compatible = []
    incompatible = []

    for model_path, spec in EMBEDDING_MODEL_SPECS.items():
        is_compat = spec['dim'] <= MAX_EMBEDDING_DIM and spec['memory_mb'] <= MAX_MEMORY_MB
        model_info = {"path": model_path, **spec, "compatible": is_compat}
        all_models.append(model_info)
        (compatible if is_compat else incompatible).append(model_info)

    return {
        "all": all_models,
        "compatible": compatible,
        "incompatible": incompatible,
        "filter_criteria": {"max_dim": MAX_EMBEDDING_DIM, "max_memory_mb": MAX_MEMORY_MB}
    }


def is_model_compatible(model_name: str) -> Tuple[bool, str]:
    """모델 호환성 검사"""
    spec = EMBEDDING_MODEL_SPECS.get(model_name)
    if spec is None:
        return True, f"알 수 없는 모델: {model_name}"

    if spec['dim'] > MAX_EMBEDDING_DIM or spec['memory_mb'] > MAX_MEMORY_MB:
        return False, f"비호환: dim={spec['dim']}, mem={spec['memory_mb']}MB"
    return True, "호환"


def filter_compatible_models() -> List[Dict]:
    """호환 가능한 모델 목록"""
    return [
        {"path": path, **spec}
        for path, spec in EMBEDDING_MODEL_SPECS.items()
        if spec['dim'] <= MAX_EMBEDDING_DIM and spec['memory_mb'] <= MAX_MEMORY_MB
    ]


# ═══════════════════════════════════════════════════════════════════════════
# 임베딩 모델
# ═══════════════════════════════════════════════════════════════════════════

def get_embedding_model(model_name: str = "intfloat/multilingual-e5-small"):
    """임베딩 모델 로드"""
    global _embed_models

    if model_name in _embed_models:
        return _embed_models[model_name]

    print(f"📦 Loading embedding model: {model_name}...")
    device = get_device()

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)
    model.eval()

    _embed_models[model_name] = (tokenizer, model)
    print(f"✅ Loaded: {model_name}")
    return tokenizer, model


def embed_text(text: str, model_name: str = "intfloat/multilingual-e5-small") -> List[float]:
    """텍스트 임베딩"""
    tokenizer, model = get_embedding_model(model_name)
    device = get_device()

    if len(text) > 1500:
        text = text[:1500]

    inputs = tokenizer(
        text, padding=True, truncation=True, max_length=512, return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    attention_mask = inputs['attention_mask']
    token_embeddings = outputs.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    embedding = (sum_embeddings / sum_mask).cpu().numpy()[0]
    return embedding.tolist()


# ═══════════════════════════════════════════════════════════════════════════
# 컬렉션 관리
# ═══════════════════════════════════════════════════════════════════════════

def list_collections() -> List[str]:
    """컬렉션 목록"""
    client = get_client()
    return [c.name for c in client.list_collections()]


def get_collection_info(collection_name: str) -> Dict:
    """컬렉션 정보"""
    try:
        client = get_client()
        collection = client.get_collection(name=collection_name)
        return {"name": collection_name, "count": collection.count()}
    except Exception:
        return {"name": collection_name, "count": 0, "error": "not found"}


def delete_collection(collection_name: str) -> bool:
    """컬렉션 삭제"""
    try:
        client = get_client()
        client.delete_collection(name=collection_name)
        return True
    except Exception:
        return False


# ═══════════════════════════════════════════════════════════════════════════
# 문서 추가
# ═══════════════════════════════════════════════════════════════════════════

def add_documents(
    texts: List[str],
    metadatas: List[Dict],
    collection_name: str = DEFAULT_COLLECTION,
    model_name: str = "intfloat/multilingual-e5-small",
) -> Dict:
    """문서 추가"""
    actual_collection_name = get_collection_name_for_model(collection_name, model_name)

    client = get_client()
    collection = client.get_or_create_collection(
        name=actual_collection_name,
        metadata={"hnsw:space": "cosine"}
    )

    embeddings = [embed_text(t, model_name) for t in texts]
    ids = [f"doc_{hashlib.md5((t + str(i)).encode()).hexdigest()[:12]}" for i, t in enumerate(texts)]

    # 메타데이터 정리 (None 값 제거, ChromaDB 호환)
    cleaned_metadatas = []
    for meta in metadatas:
        cleaned = {}
        for k, v in meta.items():
            if v is None:
                continue  # None 값 스킵
            elif isinstance(v, (str, int, float, bool)):
                cleaned[k] = v
            elif isinstance(v, list):
                # 리스트는 문자열로 변환
                cleaned[k] = str(v)
            elif isinstance(v, dict):
                # dict도 문자열로 변환
                cleaned[k] = str(v)
            else:
                cleaned[k] = str(v)
        cleaned['model'] = model_name
        cleaned_metadatas.append(cleaned)

    collection.add(
        documents=texts,
        embeddings=embeddings,
        metadatas=cleaned_metadatas,
        ids=ids,
    )

    return {
        "success": True,
        "added": len(texts),
        "collection": actual_collection_name,
    }


def add_single_text(
    text: str,
    metadata: Dict,
    collection_name: str = DEFAULT_COLLECTION,
    model_name: str = "intfloat/multilingual-e5-small",
) -> Dict:
    """단일 텍스트 추가"""
    return add_documents([text], [metadata], collection_name, model_name)


# ═══════════════════════════════════════════════════════════════════════════
# 검색 함수 (핵심 - 수정됨!)
# ═══════════════════════════════════════════════════════════════════════════

def search(
    query: str,
    collection_name: str = DEFAULT_COLLECTION,
    n_results: int = 5,
    model_name: str = "intfloat/multilingual-e5-small",
    filter_doc: Optional[str] = None,
    similarity_threshold: Optional[float] = None,
    return_low_confidence: bool = False,
) -> List[Dict]:
    """
    유사 문서 검색

    Args:
        query: 검색 쿼리
        collection_name: 컬렉션 이름
        n_results: 반환할 결과 수
        model_name: 임베딩 모델
        filter_doc: 특정 문서만 검색
        similarity_threshold: 유사도 임계값 (이하 필터링)
        return_low_confidence: True면 low confidence도 포함

    Returns:
        검색 결과 리스트
    """
    actual_collection_name = get_collection_name_for_model(collection_name, model_name)

    try:
        client = get_client()
        collection = client.get_collection(name=actual_collection_name)
    except Exception:
        return []

    if collection.count() == 0:
        return []

    query_embedding = embed_text(query, model_name)
    where_filter = {"doc_name": filter_doc} if filter_doc else None

    # 더 많이 가져와서 필터링
    fetch_count = max(n_results * 2, 10)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=fetch_count,
        where=where_filter,
        include=["documents", "metadatas", "distances"]
    )

    search_results = []
    threshold = similarity_threshold if similarity_threshold is not None else DEFAULT_SIMILARITY_THRESHOLD

    if results['documents'] and results['documents'][0]:
        for i, doc in enumerate(results['documents'][0]):
            distance = results['distances'][0][i] if results['distances'] else 0
            similarity = max(0, min(1, 1 - distance))
            confidence = calculate_confidence(similarity)

            # threshold 필터링
            if not return_low_confidence and similarity < threshold:
                continue

            metadata = results['metadatas'][0][i] if results['metadatas'] else {}
            doc_id = results['ids'][0][i] if results.get('ids') else f"doc_{i}"

            search_results.append({
                "text": doc,
                "similarity": round(similarity, 4),
                "metadata": metadata,
                "id": doc_id,
                "confidence": confidence,
            })

    # 최소 결과 보장
    if len(search_results) < MIN_RESULTS_BEFORE_FILTER and results['documents'] and results['documents'][0]:
        search_results = []
        for i, doc in enumerate(results['documents'][0][:n_results]):
            distance = results['distances'][0][i] if results['distances'] else 0
            similarity = max(0, min(1, 1 - distance))
            metadata = results['metadatas'][0][i] if results['metadatas'] else {}
            doc_id = results['ids'][0][i] if results.get('ids') else f"doc_{i}"

            search_results.append({
                "text": doc,
                "similarity": round(similarity, 4),
                "metadata": metadata,
                "id": doc_id,
                "confidence": calculate_confidence(similarity),
            })

    return search_results[:n_results]


def search_with_context(
    query: str,
    collection_name: str = DEFAULT_COLLECTION,
    n_results: int = 3,
    model_name: str = "intfloat/multilingual-e5-small",
    filter_doc: Optional[str] = None,
    similarity_threshold: Optional[float] = None,
) -> Tuple[List[Dict], str]:
    """검색 + 컨텍스트 문자열 생성"""

    results = search(
        query=query,
        collection_name=collection_name,
        n_results=n_results,
        model_name=model_name,
        filter_doc=filter_doc,
        similarity_threshold=similarity_threshold,
    )

    context_parts = []
    for i, r in enumerate(results):
        meta = r.get('metadata', {})

        # 헤더 구성 (제N조 형식)
        header_parts = []
        doc_name = meta.get('doc_name', f'문서 {i+1}')
        header_parts.append(doc_name)

        # 조항 정보 - 가독성 개선
        article_num = meta.get('article_num') or meta.get('section')
        article_type = meta.get('article_type', 'article')

        if article_num:
            if article_type == 'article':
                header_parts.append(f"제{article_num}조")
            elif article_type == 'chapter':
                header_parts.append(f"제{article_num}장")
            elif article_type == 'section':
                header_parts.append(f"제{article_num}절")
            else:
                header_parts.append(str(article_num))

        # 제목
        title = meta.get('title')
        if title and title != doc_name:
            header_parts.append(title)

        sim_str = f"{r['similarity']:.1%}"
        header = f"[{' > '.join(header_parts)}] (유사도: {sim_str})"
        context_parts.append(f"{header}\n{r['text']}")

    context = "\n\n---\n\n".join(context_parts)
    return results, context


def search_advanced(
    query: str,
    collection_name: str = DEFAULT_COLLECTION,
    n_results: int = 5,
    model_name: str = "intfloat/multilingual-e5-small",
    filter_doc: Optional[str] = None,
    similarity_threshold: Optional[float] = None,
) -> SearchResponse:
    """고급 검색 (품질 메트릭 포함)"""

    all_results = search(
        query=query,
        collection_name=collection_name,
        n_results=n_results * 2,
        model_name=model_name,
        filter_doc=filter_doc,
        similarity_threshold=0.0,
        return_low_confidence=True,
    )

    result_objects = [
        SearchResult(
            text=r['text'],
            similarity=r['similarity'],
            metadata=r['metadata'],
            id=r['id'],
            confidence=r['confidence']
        )
        for r in all_results
    ]

    threshold = similarity_threshold or DEFAULT_SIMILARITY_THRESHOLD
    filtered = [r for r in result_objects if r.similarity >= threshold]

    if len(filtered) < MIN_RESULTS_BEFORE_FILTER:
        filtered = result_objects[:MIN_RESULTS_BEFORE_FILTER]

    final_results = filtered[:n_results]

    # 품질 요약
    if final_results:
        sims = [r.similarity for r in final_results]
        quality_summary = {
            "avg_similarity": round(sum(sims) / len(sims), 4),
            "max_similarity": round(max(sims), 4),
            "min_similarity": round(min(sims), 4),
            "high_confidence_count": sum(1 for r in final_results if r.confidence == "high"),
            "threshold_used": threshold,
        }
    else:
        quality_summary = {"message": "결과 없음"}

    return SearchResponse(
        results=final_results,
        query=query,
        total_found=len(all_results),
        filtered_count=len(filtered),
        quality_summary=quality_summary
    )


# ═══════════════════════════════════════════════════════════════════════════
# 문서 삭제
# ═══════════════════════════════════════════════════════════════════════════

def delete_by_doc_name(
    doc_name: str,
    collection_name: str = DEFAULT_COLLECTION,
    model_name: Optional[str] = None
) -> Dict:
    """문서 이름으로 삭제"""
    deleted_total = 0

    if model_name:
        target_collections = [get_collection_name_for_model(collection_name, model_name)]
    else:
        target_collections = [
            col for col in list_collections()
            if col.startswith(collection_name + "__")
        ]

    for col_name in target_collections:
        try:
            client = get_client()
            collection = client.get_collection(name=col_name)
            results = collection.get(where={"doc_name": doc_name}, include=["metadatas"])
            if results['ids']:
                collection.delete(ids=results['ids'])
                deleted_total += len(results['ids'])
        except Exception:
            continue

    if deleted_total > 0:
        return {"success": True, "deleted": deleted_total}
    return {"success": False, "message": "문서를 찾을 수 없음"}


def delete_all(
    collection_name: str = DEFAULT_COLLECTION,
    model_name: Optional[str] = None
) -> Dict:
    """컬렉션 내 모든 문서 삭제"""
    try:
        if model_name:
            actual_name = get_collection_name_for_model(collection_name, model_name)
            delete_collection(actual_name)
            return {"success": True, "message": f"{actual_name} 삭제됨"}

        deleted = []
        for col_name in list_collections():
            if col_name.startswith(collection_name + "__") or col_name == collection_name:
                delete_collection(col_name)
                deleted.append(col_name)

        return {"success": True, "deleted_collections": deleted}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ═══════════════════════════════════════════════════════════════════════════
# 문서 목록
# ═══════════════════════════════════════════════════════════════════════════

def list_documents(
    collection_name: str = DEFAULT_COLLECTION,
    model_name: Optional[str] = None
) -> List[Dict]:
    """저장된 문서 목록"""
    docs = {}

    if model_name:
        target_collections = [get_collection_name_for_model(collection_name, model_name)]
    else:
        target_collections = [
            col for col in list_collections()
            if col.startswith(collection_name + "__") or col == collection_name
        ]

    for col_name in target_collections:
        try:
            client = get_client()
            collection = client.get_collection(name=col_name)
            results = collection.get(include=["metadatas"])

            for meta in (results['metadatas'] or []):
                doc_name = meta.get('doc_name', 'unknown')
                model = meta.get('model', 'unknown')
                key = f"{doc_name}|{model}"

                if key not in docs:
                    docs[key] = {
                        "doc_name": doc_name,
                        "doc_title": meta.get('doc_title'),
                        "model": model,
                        "collection": col_name,
                        "chunk_count": 0,
                        "chunk_method": meta.get('chunk_method'),
                    }
                docs[key]["chunk_count"] += 1
        except Exception:
            continue

    return list(docs.values())