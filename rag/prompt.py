"""
RAG 프롬프트 템플릿 v8.0

🔥 v8.0 개선:
- 메타데이터 기반 컨텍스트 강화
- 섹션 경로 명시로 환각 감소
- 출처 인용 강화
"""


def build_rag_prompt(query: str, context: str, language: str = "ko") -> str:
    """RAG 프롬프트 생성"""
    if language == "ko":
        return f"""당신은 GMP/QMS 규정(SOP) 전문가입니다. **모든 사고 과정(Thinking)과 최종 답변은 반드시 한국어로 작성하세요.**
아래 [참고 문서]를 바탕으로 사용자의 질문에 답변하세요.

📋 지침:
1. 반드시 제공된 문서 내용만 사용하세요. 외부 지식이나 추측은 금지입니다.
2. 답변 시 출처를 명시하세요. 예: "(SOP-00001, 5.1절 참조)"
3. 문서에 [Context: 경로] 표시가 있다면, 해당 섹션의 내용임을 인지하세요.
4. 정보를 찾을 수 없다면 "해당 문서에서 관련 정보를 찾을 수 없습니다."라고 답변하세요.
5. 여러 문서에서 정보를 찾았다면, 각 출처를 구분하여 설명하세요.

[참고 문서]
{context}

[사용자 질문]
{query}

[전문가 답변]:"""
    else:
        return f"""You are an expert in GMP/QMS regulations and SOPs. Answer based ONLY on the provided documents.

📋 Instructions:
1. Use ONLY information from the provided documents. No external knowledge or assumptions.
2. Always cite your sources. Example: "(SOP-00001, Section 5.1)"
3. If you see [Context: path], understand this indicates the section location.
4. If information is not found, say "The requested information is not available in the provided documents."
5. If multiple documents contain relevant info, distinguish each source.

[Reference Documents]
{context}

[User Question]
{query}

[Expert Answer]:"""


def build_rag_prompt_with_metadata(
    query: str, 
    chunks: list,  # List of dicts with 'text' and 'metadata'
    language: str = "ko"
) -> str:
    """
    🔥 메타데이터가 포함된 RAG 프롬프트 생성
    
    각 청크에 문서명, 섹션 경로를 명시하여 LLM이 출처를 정확히 파악
    """
    context_parts = []
    
    for i, chunk in enumerate(chunks):
        meta = chunk.get('metadata', {})
        text = chunk.get('text', '')
        
        doc_name = meta.get('sop_id') or meta.get('doc_name', '문서')
        section_path = meta.get('section_path_readable') or meta.get('section_path', '')
        
        # 청크별 컨텍스트 헤더
        header = f"📄 [{doc_name}]"
        if section_path:
            header += f" > {section_path}"
        
        context_parts.append(f"{header}\n{text}")
    
    context = "\n\n---\n\n".join(context_parts)
    
    if language == "ko":
        return f"""당신은 GMP/QMS 규정(SOP) 전문가입니다. 

📋 중요 지침:
- 각 참고 문서에는 [문서번호] > 섹션경로가 표시되어 있습니다.
- 답변 시 반드시 해당 출처를 인용하세요.
- 문서에 없는 내용은 절대 답변하지 마세요.

[참고 문서]
{context}

[사용자 질문]
{query}

[전문가 답변 - 반드시 출처 인용]:"""
    else:
        return f"""You are a GMP/QMS regulation expert.

📋 Key Instructions:
- Each reference shows [Document ID] > Section Path.
- Always cite the exact source in your answer.
- Never include information not in the documents.

[Reference Documents]
{context}

[User Question]
{query}

[Expert Answer - Must cite sources]:"""


def build_chunk_prompt(query: str, chunk_text: str, language: str = "ko") -> str:
    """단일 청크 기반 프롬프트"""
    if language == "ko":
        return f"""아래 [문서 조각]을 바탕으로 질문에 답변하세요.

지침:
- 문서 조각에 없는 내용은 답변에 포함하지 마세요.
- 정보를 찾을 수 없다면 '해당 내용에서 정보를 찾을 수 없습니다.'라고 답변하세요.

[문서 조각]
{chunk_text}

[질문]
{query}

[답변]:"""
    else:
        return f"""Answer based ONLY on the following document chunk.

[Document Chunk]
{chunk_text}

[Question]
{query}

[Answer]:"""


def build_summary_prompt(text: str, language: str = "ko") -> str:
    """요약 프롬프트"""
    if language == "ko":
        return f"""다음 문서의 핵심 내용을 요약해주세요.

[문서]
{text}

[요약]:"""
    else:
        return f"""Summarize the key points of this document.

[Document]
{text}

[Summary]:"""


def build_clarification_prompt(query: str, options: list, language: str = "ko") -> str:
    """되묻기 프롬프트"""
    options_text = "\n".join([f"- {opt}" for opt in options])

    if language == "ko":
        return f"""사용자가 "{query}"에 대해 질문했습니다.
다음 문서들이 검색되었습니다:
{options_text}

어떤 문서를 바탕으로 답변할지 정중하게 물어보세요.
한국어로 짧고 명확하게 응답하세요."""
    else:
        return f"""The user asked about "{query}".
Found documents:
{options_text}

Politely ask which document to reference.
Keep your response short and clear."""


def format_context_with_metadata(search_results: list) -> str:
    """
    🔥 검색 결과를 메타데이터와 함께 포맷팅
    
    Args:
        search_results: [{"text": "...", "metadata": {...}, "score": 0.9}, ...]
    
    Returns:
        포맷된 컨텍스트 문자열
    """
    parts = []
    
    for i, result in enumerate(search_results):
        meta = result.get('metadata', {})
        text = result.get('text', '')
        score = result.get('score', 0)
        
        # 문서 정보
        sop_id = meta.get('sop_id', '')
        section = meta.get('section_path_readable') or meta.get('section', '')
        
        # 헤더 구성
        header_parts = []
        if sop_id:
            header_parts.append(sop_id)
        if section:
            header_parts.append(section)
        
        header = " > ".join(header_parts) if header_parts else f"문서 {i+1}"
        
        # 청크 분할 정보
        chunk_part = meta.get('chunk_part')
        total_parts = meta.get('total_parts')
        if chunk_part and total_parts:
            header += f" (파트 {chunk_part}/{total_parts})"
        
        parts.append(f"📄 [{header}] (관련도: {score:.0%})\n{text}")
    
    return "\n\n---\n\n".join(parts)
