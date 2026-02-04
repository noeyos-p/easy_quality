# """
# 문서 처리 파이프라인 v8.0

# 🔥 새로운 4단계 파이프라인:
# 1. 변환: PDF, DOCX, HTML → Markdown
# 2. 분할: MarkdownHeaderTextSplitter로 헤더 기준 1차 분할
# 3. 최적화: RecursiveCharacterTextSplitter로 긴 섹션 재분할
# 4. 메타데이터: 상위 제목, 페이지 번호 등 저장
# """

# import re
# from pathlib import Path
# from typing import List, Dict, Optional, Tuple
# from dataclasses import dataclass, field
# from io import BytesIO


# # ═══════════════════════════════════════════════════════════════════════════
# # 데이터 클래스
# # ═══════════════════════════════════════════════════════════════════════════

# @dataclass
# class Chunk:
#     """청크 데이터"""
#     text: str
#     index: int = 0
#     metadata: Dict = field(default_factory=dict)


# @dataclass
# class ProcessedDocument:
#     """처리된 문서"""
#     markdown: str
#     chunks: List[Chunk]
#     metadata: Dict
    

# # ═══════════════════════════════════════════════════════════════════════════
# # 1단계: 문서 → 마크다운 변환
# # ═══════════════════════════════════════════════════════════════════════════

# def convert_to_markdown(filename: str, content: bytes) -> Tuple[str, Dict]:
#     """
#     문서를 마크다운으로 변환
    
#     Returns:
#         (markdown_text, metadata)
#     """
#     # 확장자 감지 (마지막 확장자 기준)
#     ext = filename.split('.')[-1].lower()
#     if ext == 'docx':
#         return _docx_to_markdown(filename, content)
#     elif ext == 'pdf':
#         return _pdf_to_markdown(filename, content)
#     elif ext in ['html', 'htm']:
#         return _html_to_markdown(filename, content)
#     elif ext == 'md':
#         text = content.decode('utf-8', errors='ignore')
#         return text, {"file_name": filename, "file_type": ".md"}
#     elif ext == 'txt':
#         text = content.decode('utf-8', errors='ignore')
#         return _text_to_markdown(text), {"file_name": filename, "file_type": ".txt"}
#     else:
#         # 기본: 텍스트로 처리
#         text = content.decode('utf-8', errors='ignore')
#         return text, {"file_name": filename, "file_type": "unknown"}


# def _docx_to_markdown(filename: str, content: bytes) -> Tuple[str, Dict]:
#     """DOCX → Markdown 변환"""
#     from docx import Document
#     from docx.shared import Pt
    
#     doc = Document(BytesIO(content))
#     md_lines = []
#     metadata = {"file_name": filename, "file_type": ".docx"}
    
#     # SOP ID 추출
#     sop_pattern = re.compile(r'((?:EQ-)?SOP[-_]?\d{4,5})', re.IGNORECASE)
    
#     for para in doc.paragraphs:
#         text = para.text.strip()
#         if not text:
#             md_lines.append("")
#             continue
        
#         # SOP ID 추출
#         sop_match = sop_pattern.search(text)
#         if sop_match and "sop_id" not in metadata:
#             sop_id = sop_match.group(1).upper().replace('_', '-')
#             if not sop_id.startswith('EQ-'):
#                 sop_id = 'EQ-' + sop_id
#             metadata["sop_id"] = sop_id
        
#         # 헤더 레벨 결정
#         header_level = _detect_header_level(para, text)
        
#         if header_level:
#             md_lines.append(f"{'#' * header_level} {text}")
#         else:
#             md_lines.append(text)
    
#     # 테이블 처리
#     for table in doc.tables:
#         md_lines.append("")
#         md_lines.append(_table_to_markdown(table))
    
#     return '\n'.join(md_lines), metadata


# def _detect_header_level(para, text: str) -> Optional[int]:
#     """
#     헤더 레벨 감지
    
#     Word 스타일 + 패턴 기반 감지
#     """
#     # 1. Word 스타일 기반
#     style_name = para.style.name.lower() if para.style else ""
#     if 'heading 1' in style_name or 'title' in style_name:
#         return 1
#     if 'heading 2' in style_name:
#         return 2
#     if 'heading 3' in style_name:
#         return 3
#     if 'heading 4' in style_name:
#         return 4
    
#     # 2. 패턴 기반 감지
#     # 주요 섹션 키워드 (H2)
#     main_sections = [
#         '목적', 'Purpose', '적용 범위', 'Scope', '정의', 'Definitions',
#         '책임', 'Responsibilities', '절차', 'Procedure', 
#         '참고문헌', 'Reference', '첨부', 'Attachments'
#     ]
    
#     for section in main_sections:
#         if text.startswith(section) or re.match(rf'^\d+\s+{section}', text):
#             return 2
    
#     # 숫자형 헤더
#     # 5.1.1 xxx → H4
#     if re.match(r'^\d+\.\d+\.\d+\s+', text):
#         return 4
#     # 5.1 xxx → H3
#     if re.match(r'^\d+\.\d+\s+', text):
#         return 3
#     # 5. xxx 또는 5 xxx → H2 (점 있거나 없거나, 뒤에 영문 동반 가능)
#     if re.match(r'^\d+\.?\s+([가-힣A-Za-z].+)', text):
#         return 2
    
#     # 소제목 패턴 (한글 + 영문 괄호) → H3
#     if re.match(r'^[가-힣A-Z][가-힣\s\(\)/·\-]+\s*\([A-Za-z\s&/\-:]+\)\s*$', text):
#         return 3
    
#     return None


# def _table_to_markdown(table) -> str:
#     """Word 테이블 → Markdown 테이블"""
#     rows = []
#     for row in table.rows:
#         cells = [cell.text.strip().replace('\n', ' ') for cell in row.cells]
#         rows.append(cells)
    
#     if not rows:
#         return ""
    
#     md_lines = []
#     # 헤더
#     md_lines.append("| " + " | ".join(rows[0]) + " |")
#     md_lines.append("| " + " | ".join(["---"] * len(rows[0])) + " |")
#     # 본문
#     for row in rows[1:]:
#         while len(row) < len(rows[0]):
#             row.append("")
#         md_lines.append("| " + " | ".join(row[:len(rows[0])]) + " |")
    
#     return '\n'.join(md_lines)


# def _pdf_to_markdown(filename: str, content: bytes) -> Tuple[str, Dict]:
#     """PDF → Markdown 변환"""
#     metadata = {"file_name": filename, "file_type": ".pdf"}
    
#     try:
#         # Docling 사용 (최고 품질)
#         from docling.document_converter import DocumentConverter
#         import tempfile
#         import os
        
#         with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as f:
#             f.write(content)
#             temp_path = f.name
        
#         try:
#             converter = DocumentConverter()
#             result = converter.convert(temp_path)
#             markdown = result.document.export_to_markdown()
#             metadata["parser"] = "docling"
#         finally:
#             os.unlink(temp_path)
        
#         return markdown, metadata
        
#     except ImportError:
#         pass
    
#     try:
#         # PyMuPDF (fitz) 사용
#         import fitz
        
#         pdf = fitz.open(stream=content, filetype="pdf")
#         md_lines = []
        
#         for page_num, page in enumerate(pdf):
#             text = page.get_text()
#             if text.strip():
#                 md_lines.append(f"<!-- Page {page_num + 1} -->")
#                 md_lines.append(text)
        
#         metadata["parser"] = "pymupdf"
#         return '\n'.join(md_lines), metadata
        
#     except ImportError:
#         pass
    
#     try:
#         # PyPDF2 fallback
#         from PyPDF2 import PdfReader
        
#         reader = PdfReader(BytesIO(content))
#         md_lines = []
        
#         for i, page in enumerate(reader.pages):
#             text = page.extract_text() or ''
#             if text.strip():
#                 md_lines.append(f"<!-- Page {i + 1} -->")
#                 md_lines.append(text)
        
#         metadata["parser"] = "pypdf2"
#         return '\n'.join(md_lines), metadata
        
#     except ImportError:
#         pass
    
#     try:
#         # OCR fallback (이미지 기반 PDF용)
#         md_lines, ocr_success = _pdf_to_markdown_ocr(content)
#         if ocr_success:
#             metadata["parser"] = "ocr"
#             return '\n'.join(md_lines), metadata
        
#     except ImportError:
#         pass
    
#     return "[PDF 파싱 실패: 모든 방법 시도했으나 실패]", metadata


# def _pdf_to_markdown_ocr(content: bytes) -> Tuple[List[str], bool]:
#     """
#     PDF OCR 처리 (이미지 기반 PDF용)
    
#     Returns:
#         (md_lines, success)
#     """
#     try:
#         import pytesseract
#         from pdf2image import convert_from_bytes
#         from PIL import Image
        
#         # PDF를 이미지로 변환
#         # 환경변수 PATH에 poppler가 등록되어 있어야 함
#         pages = convert_from_bytes(content, dpi=300)
        
#         md_lines = []
#         for i, page in enumerate(pages):
#             # OCR 실행 (한글+영어)
#             text = pytesseract.image_to_string(page, lang='kor+eng')
            
#             if text.strip():
#                 md_lines.append(f"<!-- Page {i + 1} (OCR) -->")
#                 md_lines.append(text)
        
#         return md_lines, len(md_lines) > 0
        
#     except Exception as e:
#         print(f"OCR 처리 실패: {e}")
#         return [], False


# def _html_to_markdown(filename: str, content: bytes) -> Tuple[str, Dict]:
#     """HTML → Markdown 변환"""
#     from bs4 import BeautifulSoup
    
#     html = content.decode('utf-8', errors='ignore')
#     soup = BeautifulSoup(html, 'html.parser')
    
#     # 불필요한 태그 제거
#     for tag in soup(['script', 'style', 'nav', 'footer', 'header']):
#         tag.decompose()
    
#     md_lines = []
    
#     # 제목
#     title = soup.title.string if soup.title else filename
#     md_lines.append(f"# {title}")
#     md_lines.append("")
    
#     # 헤더 태그 → 마크다운 헤더
#     for tag in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'li']):
#         if tag.name.startswith('h'):
#             level = int(tag.name[1])
#             md_lines.append(f"{'#' * level} {tag.get_text(strip=True)}")
#         elif tag.name == 'li':
#             md_lines.append(f"- {tag.get_text(strip=True)}")
#         else:
#             text = tag.get_text(strip=True)
#             if text:
#                 md_lines.append(text)
#         md_lines.append("")
    
#     metadata = {"file_name": filename, "file_type": ".html", "title": title}
#     return '\n'.join(md_lines), metadata


# def _text_to_markdown(text: str) -> str:
#     """
#     일반 텍스트 → 마크다운 (헤더 추론)
#     """
#     lines = text.split('\n')
#     md_lines = []
    
#     for line in lines:
#         stripped = line.strip()
#         if not stripped:
#             md_lines.append("")
#             continue
        
#         # 헤더 패턴 감지
#         # 주요 섹션
#         main_sections = ['목적', '적용 범위', '정의', '책임', '절차', '참고문헌', '첨부']
#         is_header = False
        
#         for section in main_sections:
#             if stripped.startswith(section):
#                 md_lines.append(f"## {stripped}")
#                 is_header = True
#                 break
        
#         if not is_header:
#             # 숫자 헤더
#             if re.match(r'^\d+\.\d+\.\d+\s+', stripped):
#                 md_lines.append(f"#### {stripped}")
#             elif re.match(r'^\d+\.\d+\s+', stripped):
#                 md_lines.append(f"### {stripped}")
#             elif re.match(r'^\d+\.?\s+[가-힣A-Za-z]', stripped):
#                 md_lines.append(f"## {stripped}")
#             else:
#                 md_lines.append(stripped)
    
#     return '\n'.join(md_lines)


# # ═══════════════════════════════════════════════════════════════════════════
# # 2단계: 마크다운 헤더 기준 분할
# # ═══════════════════════════════════════════════════════════════════════════

# def split_by_headers(markdown: str) -> List[Dict]:
#     """
#     마크다운 헤더 기준으로 분할
    
#     Returns:
#         [
#             {
#                 "content": "섹션 내용",
#                 "headers": {"H1": "문서 제목", "H2": "절차", "H3": "5.1 xxx"},
#                 "header_path": "절차 > 5.1 xxx"
#             },
#             ...
#         ]
#     """
#     lines = markdown.split('\n')
#     sections = []
    
#     # 현재 헤더 스택
#     current_headers = {1: None, 2: None, 3: None, 4: None, 5: None, 6: None}
#     current_content = []
    
#     def flush_section():
#         nonlocal current_content
#         if current_content:
#             content = '\n'.join(current_content).strip()
#             if content:
#                 # 헤더 경로 생성
#                 header_path_parts = []
#                 headers_dict = {}
#                 for level in range(1, 7):
#                     if current_headers[level]:
#                         headers_dict[f"H{level}"] = current_headers[level]
#                         if level >= 2:  # H2부터 경로에 포함
#                             header_path_parts.append(current_headers[level])
                
#                 sections.append({
#                     "content": content,
#                     "headers": headers_dict,
#                     "header_path": " > ".join(header_path_parts) if header_path_parts else None
#                 })
#         current_content = []
    
#     for line in lines:
#         # 헤더 감지
#         header_match = re.match(r'^(#{1,6})\s+(.+)$', line)
        
#         if header_match:
#             flush_section()
            
#             level = len(header_match.group(1))
#             header_text = header_match.group(2).strip()
            
#             # 헤더 스택 업데이트
#             current_headers[level] = header_text
#             # 하위 레벨 초기화
#             for l in range(level + 1, 7):
#                 current_headers[l] = None
            
#             # 헤더도 컨텐츠에 포함
#             current_content.append(line)
#         else:
#             current_content.append(line)
    
#     flush_section()
#     return sections


# # ═══════════════════════════════════════════════════════════════════════════
# # 3단계: 긴 섹션 재분할 (RecursiveCharacterTextSplitter 스타일)
# # ═══════════════════════════════════════════════════════════════════════════

# def split_recursive(
#     text: str,
#     chunk_size: int = 500,
#     chunk_overlap: int = 50,
#     separators: List[str] = None
# ) -> List[str]:
#     """
#     RecursiveCharacterTextSplitter 스타일 분할
    
#     🔥 v8.1 개선:
#     - 테이블 행 단위 분할 지원 ("\n| ")
#     - 테이블 구조 보존
#     - 테이블 내부에서는 overlap 비활성화
#     """
#     if separators is None:
#         # 🔥 테이블 행 구분자 추가 (표 한복판 분할 방지)
#         separators = [
#             "\n\n",      # 문단 구분
#             "\n| ",      # 🔥 마크다운 테이블 행 (표 보존)
#             "\n",        # 줄바꿈
#             ". ",        # 문장
#             "。",        # 한국어/일본어 문장
#             " ",         # 단어
#             ""           # 문자
#         ]
    
#     if len(text) <= chunk_size:
#         return [text]
    
#     # 🔥 테이블인지 확인 (테이블이면 overlap 비활성화)
#     is_table = text.strip().startswith('|') or '\n|' in text
#     effective_overlap = 0 if is_table else chunk_overlap
    
#     chunks = []
    
#     # 적절한 구분자 찾기
#     for sep in separators:
#         if sep in text:
#             parts = text.split(sep)
#             current_chunk = ""
            
#             for part in parts:
#                 if len(current_chunk) + len(part) + len(sep) <= chunk_size:
#                     if current_chunk:
#                         current_chunk += sep + part
#                     else:
#                         current_chunk = part
#                 else:
#                     if current_chunk:
#                         chunks.append(current_chunk)
                    
#                     # 파트가 너무 크면 재귀 분할
#                     if len(part) > chunk_size:
#                         sub_chunks = split_recursive(
#                             part, chunk_size, chunk_overlap,
#                             separators[separators.index(sep) + 1:]
#                         )
#                         chunks.extend(sub_chunks)
#                         current_chunk = ""
#                     else:
#                         current_chunk = part
            
#             if current_chunk:
#                 chunks.append(current_chunk)
            
#             # 오버랩 적용 (테이블이 아닐 때만)
#             if effective_overlap > 0 and len(chunks) > 1:
#                 chunks = _apply_overlap(chunks, effective_overlap)
            
#             return chunks
    
#     # 구분자가 없으면 강제 분할
#     step = chunk_size - effective_overlap if effective_overlap > 0 else chunk_size
#     return [text[i:i+chunk_size] for i in range(0, len(text), step)]


# def _apply_overlap(chunks: List[str], overlap: int) -> List[str]:
#     """청크 간 오버랩 적용"""
#     if len(chunks) <= 1:
#         return chunks
    
#     result = [chunks[0]]
#     for i in range(1, len(chunks)):
#         prev_end = chunks[i-1][-overlap:] if len(chunks[i-1]) > overlap else chunks[i-1]
#         result.append(prev_end + chunks[i])
    
#     return result


# # ═══════════════════════════════════════════════════════════════════════════
# # 4단계: 청크 생성 및 메타데이터 추가
# # ═══════════════════════════════════════════════════════════════════════════

# def create_chunks(
#     sections: List[Dict],
#     metadata: Dict,
#     chunk_size: int = 500,
#     chunk_overlap: int = 50,
#     add_context_prefix: bool = True  # 🔥 컨텍스트 프리픽스 추가 옵션
# ) -> List[Chunk]:
#     """
#     섹션을 청크로 변환하고 메타데이터 추가
    
#     🔥 v8.1 개선:
#     - add_context_prefix: 재분할된 청크에 헤더 경로를 텍스트로 삽입
#       → 검색 성능 향상, 환각 감소
#     """
#     chunks = []
#     idx = 0
    
#     sop_id = metadata.get("sop_id")
#     doc_name = metadata.get("file_name")
    
#     for section in sections:
#         content = section["content"]
#         headers = section.get("headers", {})
#         header_path = section.get("header_path")
        
#         # 섹션이 너무 길면 재분할
#         if len(content) > chunk_size:
#             text_chunks = split_recursive(content, chunk_size, chunk_overlap)
#             is_split = len(text_chunks) > 1
#         else:
#             text_chunks = [content]
#             is_split = False
        
#         for i, text in enumerate(text_chunks):
#             if not text.strip():
#                 continue
            
#             # 🔥 재분할된 청크(2번째부터)에 컨텍스트 프리픽스 추가
#             if add_context_prefix and is_split and i > 0 and header_path:
#                 context_prefix = f"[Context: {header_path}]\n\n"
#                 text = context_prefix + text
            
#             # 섹션 타입 결정
#             section_type = "text"
#             section_num = None
            
#             if headers.get("H4"):
#                 section_type = "subsubsection"
#                 match = re.match(r'^(\d+\.\d+\.\d+)', headers["H4"])
#                 if match:
#                     section_num = match.group(1)
#             elif headers.get("H3"):
#                 section_type = "subsection"
#                 match = re.match(r'^(\d+\.\d+)', headers["H3"])
#                 if match:
#                     section_num = match.group(1)
#             elif headers.get("H2"):
#                 section_type = "section"
#                 match = re.match(r'^(\d+)', headers["H2"])
#                 if match:
#                     section_num = match.group(1)
            
#             # 가장 낮은 레벨 헤더를 section으로 사용
#             section_display = headers.get("H4") or headers.get("H3") or headers.get("H2") or headers.get("H1")
            
#             chunks.append(Chunk(
#                 text=text.strip(),
#                 index=idx,
#                 metadata={
#                     "doc_name": doc_name,
#                     "doc_title": sop_id or doc_name,
#                     "sop_id": sop_id,
#                     "article_num": section_num,
#                     "article_type": section_type,
#                     "section": section_display,
#                     "section_path": header_path,
#                     "section_path_readable": header_path,
#                     "H1": headers.get("H1"),
#                     "H2": headers.get("H2"),
#                     "H3": headers.get("H3"),
#                     "H4": headers.get("H4"),
#                     "chunk_part": i + 1 if is_split else None,  # 🔥 분할된 청크 번호
#                     "total_parts": len(text_chunks) if is_split else None,
#                 }
#             ))
#             idx += 1
    
#     return chunks


# # ═══════════════════════════════════════════════════════════════════════════
# # 메인 파이프라인
# # ═══════════════════════════════════════════════════════════════════════════

# def process_document(
#     filename: str,
#     content: bytes,
#     chunk_size: int = 500,
#     chunk_overlap: int = 50,
#     debug: bool = False
# ) -> ProcessedDocument:
#     """
#     문서 처리 메인 파이프라인
    
#     1. 변환: 문서 → 마크다운
#     2. 분할: 헤더 기준 1차 분할
#     3. 최적화: 긴 섹션 재분할
#     4. 메타데이터: 청크 생성
#     """
#     # 1단계: 마크다운 변환
#     markdown, metadata = convert_to_markdown(filename, content)
    
#     if debug:
#         print(f"\n{'='*60}")
#         print(f"📄 1단계: 마크다운 변환")
#         print(f"   파일: {filename}")
#         print(f"   마크다운 길이: {len(markdown)}")
#         print(f"   메타데이터: {metadata}")
#         print(f"\n   마크다운 미리보기 (처음 500자):")
#         print(f"   {'-'*50}")
#         for line in markdown[:500].split('\n'):
#             print(f"   {line}")
    
#     # 2단계: 헤더 기준 분할
#     sections = split_by_headers(markdown)
    
#     if debug:
#         print(f"\n{'='*60}")
#         print(f"📄 2단계: 헤더 기준 분할")
#         print(f"   섹션 수: {len(sections)}")
#         print(f"\n   섹션 목록:")
#         for i, sec in enumerate(sections[:10]):
#             path = sec.get('header_path', 'N/A')
#             content_preview = sec['content'][:50].replace('\n', ' ')
#             print(f"   [{i}] 📍 {path}")
#             print(f"       {content_preview}...")
    
#     # 3단계 & 4단계: 청크 생성
#     chunks = create_chunks(sections, metadata, chunk_size, chunk_overlap)
    
#     if debug:
#         print(f"\n{'='*60}")
#         print(f"📄 3-4단계: 청크 생성")
#         print(f"   총 청크 수: {len(chunks)}")
#         print(f"\n   청크 샘플:")
#         for chunk in chunks[:5]:
#             path = chunk.metadata.get('section_path_readable', 'N/A')
#             print(f"   [{chunk.index}] 📍 {path}")
#             print(f"       {chunk.text[:60]}...")
    
#     return ProcessedDocument(
#         markdown=markdown,
#         chunks=chunks,
#         metadata=metadata
#     )


# # ═══════════════════════════════════════════════════════════════════════════
# # 테스트
# # ═══════════════════════════════════════════════════════════════════════════

# if __name__ == "__main__":
#     # 테스트 마크다운
#     test_md = """# EQ-SOP-00010 품질관리기준서

# ## 1 목적 Purpose

# 본 기준서는 품질관리기준서의 작성, 검토, 승인에 관한 기준을 정한다.

# ## 2 적용 범위 Scope

# 본 기준서는 회사 내 품질관리 활동 전반에 적용된다.

# ## 5 절차 Procedure

# ### 5.1 품질관리기준서의 구성 및 관리

# 품질관리기준서는 다음 항목을 포함한다.

# #### 5.1.1 문서번호 체계

# 문서번호는 EQ-SOP-XXXXX 형식을 따른다.

# #### 5.1.2 개정 관리

# 개정 시 변경 이력을 기록한다.

# ### 5.2 시험방법서와의 연계

# 시험방법서는 품질관리기준서와 연계되어야 한다.
# """
    
#     # 파이프라인 테스트
#     result = process_document(
#         "test.md",
#         test_md.encode('utf-8'),
#         chunk_size=300,
#         debug=True
#     )
    
#     print(f"\n{'='*60}")
#     print(f"✅ 최종 결과")
#     print(f"   총 청크: {len(result.chunks)}")
