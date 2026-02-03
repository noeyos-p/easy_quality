"""
RAG 패키지 v6.3
- section_path 계층 추적
- intro 블록 RAG 제외 (품질 향상)
- doc_title SOP ID 기반
"""

# document_loader, chunker 제거됨


from .vector_store import (
    search,
    search_with_context,
    search_advanced,
    add_documents,
    add_single_text,
    list_documents,
    list_collections,
    delete_by_doc_name,
    delete_all,
    get_embedding_model_info,
    filter_compatible_models,
    is_model_compatible,
    get_collection_info,
    EMBEDDING_MODEL_SPECS,
    MAX_EMBEDDING_DIM,
    MAX_MEMORY_MB,
    DEFAULT_SIMILARITY_THRESHOLD,
    HIGH_CONFIDENCE_THRESHOLD,
    SearchResult,
    SearchResponse,
)

from .llm import (
    get_llm_response,
    OllamaLLM,
    analyze_search_results,
    generate_clarification_question,
    OLLAMA_MODELS,
    HUGGINGFACE_MODELS,
)

# prompt 제거됨

__version__ = "6.3.0"