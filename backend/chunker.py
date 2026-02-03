"""
텍스트 청킹 모듈 v6.2 - section_path 지원
- sentence: 문장 단위
- paragraph: 문단 단위
- article: 조항 단위 (SOP/법률)
- recursive: RecursiveCharacterTextSplitter
- semantic: 의미 기반 분할
- llm: LLM 기반 구조 파싱

🔥 v6.2 추가:
- section_path: "5 > 5.1 > 5.1.1" 형태의 계층 경로 지원
- section_path_readable: 사람이 읽기 쉬운 형태
"""

import re
from typing import List, Optional, Callable, Dict
from dataclasses import dataclass, field
import numpy as np


@dataclass
class Chunk:
    """청크 데이터"""
    text: str
    index: int = 0
    metadata: Dict = field(default_factory=dict)


CHUNK_METHODS = {
    "sentence": "문장 단위",
    "paragraph": "문단 단위",
    "article": "조항 단위 (SOP/법률)",
    "recursive": "Recursive (랭체인 스타일)",
    "semantic": "의미 기반",
    "llm": "LLM 파싱",
}


# ═══════════════════════════════════════════════════════════════════════════
# 조항 패턴
# ═══════════════════════════════════════════════════════════════════════════

def normalize_section_number(text: str) -> str:
    """
    섹션 번호 정규화 (document_loader.py와 동일)
    - "5-1" → "5.1"
    - "5．1" → "5.1" (전각 점)
    - "Ⅴ" → "5" (로마 숫자)
    """
    # 로마 숫자 → 아라비아 숫자
    roman_map = {
        'Ⅰ': '1', 'Ⅱ': '2', 'Ⅲ': '3', 'Ⅳ': '4', 'Ⅴ': '5',
        'Ⅵ': '6', 'Ⅶ': '7', 'Ⅷ': '8', 'Ⅸ': '9', 'Ⅹ': '10',
    }
    for roman, arabic in roman_map.items():
        text = text.replace(roman, arabic)
    
    # 전각 문자 → 반각
    text = text.replace('．', '.').replace('－', '-').replace('　', ' ')
    
    # 하이픈을 점으로 (5-1 → 5.1)
    text = re.sub(r'(\d+)\s*[-‐‑–—]\s*(\d+)', r'\1.\2', text)
    
    # 숫자 사이 공백 제거 (5 . 1 → 5.1)
    text = re.sub(r'(\d+)\s*\.\s*(\d+)', r'\1.\2', text)
    
    return text


# 🔥 구체적인 것부터 매칭! (순서 중요)
ARTICLE_PATTERNS = [
    # 숫자형: 가장 구체적인 것부터!
    (r'^(\d+\.\d+\.\d+)\s+(.+)', 'subsubsection'),      # 5.1.1 xxx
    (r'^(\d+\.\d+)\s+(.+)', 'subsection'),              # 5.1 xxx  
    (r'^(\d+)\.\s+(.+)', 'section'),                    # 5. xxx (점 있음)
    (r'^(\d+)\s+([가-힣A-Za-z].+)', 'section'),         # 5 xxx (점 없음)
    
    # 한글 조항
    (r'^제\s*(\d+)\s*조\s*(.*)', 'article'),
    (r'^제\s*(\d+)\s*장\s*(.*)', 'chapter'),
    (r'^제\s*(\d+)\s*절\s*(.*)', 'section'),
    (r'^제\s*(\d+)\s*레벨\s*[:\(]?\s*(.+)', 'level'),
]

# 이름 기반 섹션 (숫자 없을 때만)
NAMED_SECTION_KEYWORDS = ['목적', '적용범위', '적용 범위', '정의', '책임', '절차', '참고문헌', '첨부']


def detect_article(line: str) -> Optional[tuple]:
    """조항 감지 (정규화 후 매칭)"""
    line = normalize_section_number(line.strip())
    
    # 숫자 패턴 먼저
    for pattern, a_type in ARTICLE_PATTERNS:
        match = re.match(pattern, line)
        if match:
            num = match.group(1)
            title = match.group(2).strip() if match.lastindex >= 2 else ""
            return (num, a_type, title)
    
    # 이름 기반 섹션 (숫자로 시작하지 않을 때만)
    if not re.match(r'^\d', line):
        for keyword in NAMED_SECTION_KEYWORDS:
            if line.startswith(keyword):
                return (keyword, 'named_section', '')
    
    return None


# ═══════════════════════════════════════════════════════════════════════════
# 1. 문장 단위
# ═══════════════════════════════════════════════════════════════════════════

def split_by_sentences(text: str, max_length: int = 300, overlap: int = 50) -> List[str]:
    """문장 단위 분할"""
    sentences = re.split(r'(?<=[.!?。])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sent_len = len(sentence)

        if current_length + sent_len > max_length and current_chunk:
            chunks.append(' '.join(current_chunk))

            # 오버랩
            overlap_sentences = []
            temp_len = 0
            for s in reversed(current_chunk):
                if temp_len + len(s) <= overlap:
                    overlap_sentences.insert(0, s)
                    temp_len += len(s)
                else:
                    break
            current_chunk = overlap_sentences
            current_length = sum(len(s) for s in current_chunk)

        current_chunk.append(sentence)
        current_length += sent_len

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks


# ═══════════════════════════════════════════════════════════════════════════
# 2. 문단 단위
# ═══════════════════════════════════════════════════════════════════════════

def split_by_paragraphs(text: str, max_length: int = 1000) -> List[str]:
    """문단 단위 분할"""
    paragraphs = re.split(r'\n\s*\n', text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    chunks = []
    current_chunk = []
    current_length = 0

    for para in paragraphs:
        para_len = len(para)

        if para_len > max_length:
            if current_chunk:
                chunks.append('\n\n'.join(current_chunk))
                current_chunk = []
                current_length = 0
            chunks.extend(split_by_sentences(para, max_length))

        elif current_length + para_len > max_length:
            chunks.append('\n\n'.join(current_chunk))
            current_chunk = [para]
            current_length = para_len

        else:
            current_chunk.append(para)
            current_length += para_len

    if current_chunk:
        chunks.append('\n\n'.join(current_chunk))

    return chunks


# ═══════════════════════════════════════════════════════════════════════════
# 3. 조항 단위
# ═══════════════════════════════════════════════════════════════════════════

def split_by_articles(text: str, max_length: int = 500, overlap: int = 50) -> List[dict]:
    """조항 단위 분할"""
    lines = text.split('\n')
    articles = []

    current = {'lines': [], 'article_num': None, 'article_type': 'intro'}

    def flush():
        if current['lines']:
            article_text = '\n'.join(current['lines']).strip()
            if article_text:
                articles.append({
                    'text': article_text,
                    'article_num': current['article_num'],
                    'article_type': current['article_type'],
                })

    for line in lines:
        detected = detect_article(line)
        if detected:
            flush()
            current = {
                'lines': [line],
                'article_num': detected[0],
                'article_type': detected[1],
            }
        else:
            current['lines'].append(line)

    flush()

    # 긴 조항 분할
    result = []
    for art in articles:
        if len(art['text']) > max_length:
            sub_texts = split_by_sentences(art['text'], max_length, overlap)
            for i, sub in enumerate(sub_texts):
                result.append({
                    **art,
                    'text': sub,
                    'chunk_part': f"{i+1}/{len(sub_texts)}"
                })
        else:
            result.append(art)

    return result


# ═══════════════════════════════════════════════════════════════════════════
# 4. Recursive Character Text Splitter
# ═══════════════════════════════════════════════════════════════════════════

class RecursiveCharacterTextSplitter:
    """랭체인 스타일 Recursive 분할기"""

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        separators: List[str] = None
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]

    def split_text(self, text: str) -> List[str]:
        """텍스트 분할"""
        return self._split_recursive(text, self.separators)

    def _split_recursive(self, text: str, separators: List[str]) -> List[str]:
        if not text:
            return []

        if len(text) <= self.chunk_size:
            return [text]

        if not separators:
            return self._split_by_size(text)

        sep = separators[0]
        remaining_seps = separators[1:]

        if sep:
            parts = text.split(sep)
        else:
            return self._split_by_size(text)

        chunks = []
        current_chunk = []
        current_size = 0

        for part in parts:
            part_with_sep = part + sep if sep else part
            part_size = len(part_with_sep)

            if part_size > self.chunk_size:
                if current_chunk:
                    chunks.append(sep.join(current_chunk))
                    current_chunk = []
                    current_size = 0
                chunks.extend(self._split_recursive(part, remaining_seps))

            elif current_size + part_size > self.chunk_size:
                if current_chunk:
                    chunks.append(sep.join(current_chunk))

                    # 오버랩
                    overlap_parts = []
                    overlap_size = 0
                    for p in reversed(current_chunk):
                        if overlap_size + len(p) <= self.chunk_overlap:
                            overlap_parts.insert(0, p)
                            overlap_size += len(p)
                        else:
                            break
                    current_chunk = overlap_parts
                    current_size = sum(len(p) for p in current_chunk)

                current_chunk.append(part)
                current_size += part_size
            else:
                current_chunk.append(part)
                current_size += part_size

        if current_chunk:
            chunks.append(sep.join(current_chunk))

        return chunks

    def _split_by_size(self, text: str) -> List[str]:
        """크기 기반 분할"""
        chunks = []
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            chunks.append(text[i:i + self.chunk_size])
        return chunks


def split_recursive(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Recursive 분할 헬퍼"""
    splitter = RecursiveCharacterTextSplitter(chunk_size, overlap)
    return splitter.split_text(text)


# ═══════════════════════════════════════════════════════════════════════════
# 5. Semantic 분할
# ═══════════════════════════════════════════════════════════════════════════

class SemanticSplitter:
    """의미 기반 분할기"""

    def __init__(
        self,
        embed_function: Callable[[str], List[float]],
        threshold: float = 0.5,
        max_chunk_size: int = 500
    ):
        self.embed_function = embed_function
        self.threshold = threshold
        self.max_chunk_size = max_chunk_size

    def split_text(self, text: str) -> List[str]:
        """의미 기반 분할"""
        sentences = re.split(r'(?<=[.!?。])\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if len(sentences) <= 1:
            return sentences

        # 임베딩 계산
        embeddings = [self.embed_function(s) for s in sentences]

        # 유사도 기반 분할점 찾기
        split_points = [0]
        for i in range(1, len(embeddings)):
            sim = self._cosine_similarity(embeddings[i-1], embeddings[i])
            if sim < self.threshold:
                split_points.append(i)
        split_points.append(len(sentences))

        # 청크 생성
        chunks = []
        for i in range(len(split_points) - 1):
            start, end = split_points[i], split_points[i+1]
            chunk_text = ' '.join(sentences[start:end])

            if len(chunk_text) > self.max_chunk_size:
                chunks.extend(split_recursive(chunk_text, self.max_chunk_size))
            else:
                chunks.append(chunk_text)

        return chunks

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        a, b = np.array(a), np.array(b)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def split_semantic(
    text: str,
    embed_function: Callable = None,
    threshold: float = 0.5,
    max_chunk_size: int = 500
) -> List[str]:
    """Semantic 분할 헬퍼"""
    if embed_function is None:
        return split_recursive(text, max_chunk_size)

    splitter = SemanticSplitter(embed_function, threshold, max_chunk_size)
    return splitter.split_text(text)


# ═══════════════════════════════════════════════════════════════════════════
# 6. LLM 기반 파싱
# ═══════════════════════════════════════════════════════════════════════════

LLM_PARSING_PROMPT = """다음 문서의 논리적 구조를 분석하고 JSON으로 반환하세요.

문서:
{text}

JSON 형식 (배열만 반환):
[{{"title": "섹션제목", "start": 시작인덱스, "end": 끝인덱스}}, ...]"""


def split_by_llm(
    text: str,
    llm_function: Callable[[str], str] = None,
    max_chunk_size: int = 500,
    fallback_method: str = "recursive"
) -> List[dict]:
    """LLM 기반 분할"""
    if llm_function is None:
        return _fallback_split(text, max_chunk_size, fallback_method)

    analysis_text = text[:8000] if len(text) > 8000 else text

    try:
        import json

        prompt = LLM_PARSING_PROMPT.format(text=analysis_text)
        response = llm_function(prompt)

        json_match = re.search(r'\[.*\]', response, re.DOTALL)
        if not json_match:
            raise ValueError("JSON not found")

        sections = json.loads(json_match.group())

        chunks = []
        for i, section in enumerate(sections):
            start = section.get('start', 0)
            end = section.get('end', len(text))
            section_text = text[start:end].strip()

            if len(section_text) > max_chunk_size:
                sub_chunks = split_recursive(section_text, max_chunk_size)
                for j, sub in enumerate(sub_chunks):
                    chunks.append({
                        'text': sub,
                        'title': section.get('title', f'Section {i+1}'),
                        'section_index': i,
                        'sub_index': j,
                        'parse_method': 'llm'
                    })
            else:
                chunks.append({
                    'text': section_text,
                    'title': section.get('title', f'Section {i+1}'),
                    'section_index': i,
                    'parse_method': 'llm'
                })

        return chunks if chunks else _fallback_split(text, max_chunk_size, fallback_method)

    except Exception as e:
        print(f"⚠️ LLM 파싱 실패: {e}")
        return _fallback_split(text, max_chunk_size, fallback_method)


def _fallback_split(text: str, max_size: int, method: str) -> List[dict]:
    """Fallback 분할"""
    if method == "recursive":
        raw = split_recursive(text, max_size)
    elif method == "sentence":
        raw = split_by_sentences(text, max_size)
    else:
        raw = split_by_paragraphs(text, max_size)

    return [{'text': c, 'parse_method': f'fallback_{method}'} for c in raw]


# ═══════════════════════════════════════════════════════════════════════════
# 통합 인터페이스
# ═══════════════════════════════════════════════════════════════════════════

def create_chunks(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
    method: str = "sentence",
    embed_function: Callable = None,
    llm_function: Callable = None,
    semantic_threshold: float = 0.5,
) -> List[Chunk]:
    """통합 청킹 함수"""

    if method == "article":
        raw = split_by_articles(text, chunk_size, overlap)
        return [
            Chunk(
                text=r['text'].strip(),
                index=i,
                metadata={
                    "article_num": r.get('article_num'),
                    "article_type": r.get('article_type'),
                    "chunk_part": r.get('chunk_part'),
                }
            )
            for i, r in enumerate(raw) if r['text'].strip()
        ]

    elif method == "recursive":
        raw = split_recursive(text, chunk_size, overlap)

    elif method == "semantic":
        raw = split_semantic(text, embed_function, semantic_threshold, chunk_size)

    elif method == "llm":
        raw_data = split_by_llm(text, llm_function, chunk_size)
        return [
            Chunk(
                text=r['text'].strip(),
                index=i,
                metadata={"title": r.get('title'), "parse_method": r.get('parse_method')}
            )
            for i, r in enumerate(raw_data) if r['text'].strip()
        ]

    elif method == "paragraph":
        raw = split_by_paragraphs(text, chunk_size)

    else:  # sentence
        raw = split_by_sentences(text, chunk_size, overlap)

    return [
        Chunk(text=c.strip(), index=i, metadata={})
        for i, c in enumerate(raw) if c.strip()
    ]


def create_chunks_from_blocks(
    doc,  # ParsedDocument
    chunk_size: int = 500,
    overlap: int = 50,
    method: str = "recursive",
    exclude_intro: bool = True,           # 🔥 intro 블록 제외 옵션
    require_section_path: bool = False,   # 🔥 section_path 필수 여부
) -> List[Chunk]:
    """
    블록 기반 청킹 (메타데이터 유지, 섹션별 SOP ID)
    
    🔥 v6.3 개선:
    - intro 블록 제거 (RAG 품질 향상)
    - section_path 없는 블록 선택적 제거
    - doc_title을 SOP ID 기반으로 설정
    """
    chunks = []
    idx = 0
    
    # 🔥 doc_title을 SOP ID 기반으로 설정
    sop_id_global = doc.metadata.get("sop_id")
    doc_title_base = doc.metadata.get("title", "")
    
    if sop_id_global:
        # SOP ID가 있으면 우선 사용
        doc_title = sop_id_global
        # 제목도 있으면 조합
        if doc_title_base and doc_title_base != sop_id_global:
            doc_title = f"{doc_title_base} ({sop_id_global})"
    else:
        doc_title = doc_title_base or doc.metadata.get("file_name", "문서")

    for block in doc.blocks:
        # 🔥 1. intro, toc 블록 제거 (RAG에서 제외)
        if exclude_intro and block.metadata.get("article_type") in ["intro", "toc"]:
            continue
        
        # 🔥 2. section_path 없는 블록 제거 (선택적)
        if require_section_path and not block.metadata.get("section_path"):
            continue
        
        if method == "recursive":
            texts = split_recursive(block.text, chunk_size, overlap)
        else:
            texts = [block.text]

        for t in texts:
            if not t.strip():
                continue

            # 메타데이터 구성
            article_num = block.metadata.get('article_num') or block.section
            article_type = block.metadata.get('article_type', 'article')
            
            # 블록별 SOP ID 우선, 없으면 문서 전체 SOP ID
            sop_id = block.metadata.get('sop_id') or sop_id_global

            # 가독성 좋은 section 표시
            section_display = None
            if article_num:
                if article_type == 'article':
                    section_display = f"제{article_num}조"
                elif article_type == 'chapter':
                    section_display = f"제{article_num}장"
                elif article_type == 'section':
                    section_display = article_num  # "1", "6" 등
                elif article_type == 'subsection':
                    section_display = article_num  # "6.1", "6.2" 등
                elif article_type == 'subsubsection':
                    section_display = article_num  # "5.1.1" 등
                elif article_type == 'level':
                    section_display = f"Level {article_num}"  # "Level 1" 등
                elif article_type == 'named_section':
                    section_display = article_num  # "목적", "절차" 등
                else:
                    section_display = str(article_num)

            # section_path
            section_path = block.metadata.get("section_path")
            section_path_readable = block.metadata.get("section_path_readable")

            chunks.append(Chunk(
                text=t.strip(),
                index=idx,
                metadata={
                    "doc_name": doc.metadata.get("file_name"),
                    "doc_title": doc_title,  # 🔥 SOP ID 기반 doc_title
                    "sop_id": sop_id,
                    "version": doc.metadata.get("version"),
                    "article_num": article_num,
                    "article_type": article_type,
                    "section": section_display,
                    "section_path": section_path,
                    "section_path_readable": section_path_readable,
                    "title": block.metadata.get("title"),  # 조항 제목
                    "page": block.page,
                    "block_type": block.block_type,
                }
            ))
            idx += 1

    return chunks


def get_available_methods() -> dict:
    """사용 가능한 청킹 방법"""
    return CHUNK_METHODS.copy()
