"""
문서 로더 v7.0 - 완전 리팩토링

🔥 핵심 개선:
1. 문서 형식 자동 감지 (숫자형 vs 이름형)
2. 목차 감지 및 스킵
3. 정규화 파이프라인
4. 계층 스택 단순화
"""

import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from io import BytesIO


# ═══════════════════════════════════════════════════════════════════════════
# 데이터 클래스
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ContentBlock:
    """문서 블록"""
    text: str
    block_type: str = "text"
    section: Optional[str] = None
    page: Optional[int] = None
    metadata: Dict = field(default_factory=dict)


@dataclass 
class ParsedDocument:
    """파싱된 문서"""
    text: str
    blocks: List[ContentBlock]
    metadata: Dict
    tables: List[Dict] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════
# 텍스트 정규화
# ═══════════════════════════════════════════════════════════════════════════

def normalize_text(text: str) -> str:
    """
    텍스트 정규화
    - 전각 → 반각
    - 로마 숫자 → 아라비아 숫자
    - 섹션 번호 형식 통일
    """
    # 로마 숫자 변환
    roman_map = {
        'Ⅰ': '1', 'Ⅱ': '2', 'Ⅲ': '3', 'Ⅳ': '4', 'Ⅴ': '5',
        'Ⅵ': '6', 'Ⅶ': '7', 'Ⅷ': '8', 'Ⅸ': '9', 'Ⅹ': '10',
        'ⅰ': '1', 'ⅱ': '2', 'ⅲ': '3', 'ⅳ': '4', 'ⅴ': '5',
    }
    for roman, arabic in roman_map.items():
        text = text.replace(roman, arabic)
    
    # 전각 → 반각
    text = text.replace('．', '.').replace('－', '-').replace('　', ' ')
    text = text.replace('：', ':').replace('（', '(').replace('）', ')')
    
    # 하이픈 → 점 (5-1 → 5.1)
    text = re.sub(r'(\d+)\s*[-‐‑–—]\s*(\d+)', r'\1.\2', text)
    
    # 숫자.숫자 공백 제거 (5 . 1 → 5.1)
    text = re.sub(r'(\d+)\s*\.\s*(\d+)', r'\1.\2', text)
    
    return text


# ═══════════════════════════════════════════════════════════════════════════
# 섹션 패턴 매칭
# ═══════════════════════════════════════════════════════════════════════════

# 주요 섹션 키워드 (한글 + 영문)
SECTION_KEYWORDS = {
    '목적': 'Purpose',
    '적용범위': 'Scope',
    '적용 범위': 'Scope',
    '정의': 'Definitions',
    '책임': 'Responsibilities',
    '절차': 'Procedure',
    '참고문헌': 'Reference',
    '첨부': 'Attachments',
    '기타': 'Others',
    '목차': 'Table of Contents',
}


def detect_section(line: str) -> Optional[Dict]:
    """
    라인에서 섹션 정보 추출
    
    Returns:
        {
            "num": "5.1" 또는 "목적",
            "type": "section" | "subsection" | "subsubsection" | "named_section" | "toc",
            "title": "제품표준서 번호 체계...",
            "level": 1 | 2 | 3  # 계층 레벨
        }
    """
    line = normalize_text(line.strip())
    if not line:
        return None
    
    # 1️⃣ 목차 감지
    if line.startswith('목차') or line.lower().startswith('table of contents'):
        return {"num": "TOC", "type": "toc", "title": "", "level": 0}
    
    # 2️⃣ 숫자형 섹션 (가장 구체적인 것부터!)
    patterns = [
        # 5.1.1 형식
        (r'^(\d+\.\d+\.\d+)\s+(.+)', 'subsubsection', 3),
        # 5.1 형식  
        (r'^(\d+\.\d+)\s+(.+)', 'subsection', 2),
        # 5. 목적 Purpose 형식 (점 있거나 없거나, 뒤에 영문 동반 가능)
        (r'^(\d+)\.?\s+([가-힣A-Za-z].+)', 'section', 1),
        # 제N조, 제N장
        (r'^제\s*(\d+)\s*조\s*(.*)', 'article', 1),
        (r'^제\s*(\d+)\s*장\s*(.*)', 'chapter', 1),
        (r'^제\s*(\d+)\s*레벨\s*[:\(]?\s*(.+)', 'level', 2),
    ]
    
    for pattern, sec_type, level in patterns:
        m = re.match(pattern, line)
        if m:
            num = m.group(1)
            title = m.group(2).strip() if m.lastindex >= 2 else ""
            return {"num": num, "type": sec_type, "title": title, "level": level}
    
    # 3️⃣ 이름형 섹션 (숫자로 시작하지 않을 때만!)
    if re.match(r'^\d', line):
        return None  # 숫자로 시작하면 이름형 아님
    
    # 주요 섹션 키워드
    for keyword, eng in SECTION_KEYWORDS.items():
        if keyword == '목차':
            continue  # 이미 위에서 처리
        
        # "목적 Purpose" 또는 "목적" 형식 (영문 명칭이 포함되거나 포함되지 않은 경우 모두 대응)
        pattern = rf'^{re.escape(keyword)}(\s+.*{re.escape(eng)}.*)?'
        if re.match(pattern, line, re.IGNORECASE):
            return {"num": keyword, "type": "named_section", "title": eng, "level": 1}
    
    # 4️⃣ 소제목 감지 (한글 또는 영문 + 영문 괄호로 끝나는 경우)
    # 예: "제품표준서 번호 체계 및 문서 유형 (Numbering & Document Type)"
    # 예: "검토 및 승인 (Review & Approval)"
    # 예: "제정(작성) 및 등록 (Creation & Registration)"
    # 예: "EDMS 계정 및 권한관리 (Account & Role Management)"
    
    # 패턴 1: 한글로 시작
    subtitle_pattern1 = r'^([가-힣][가-힣\s\(\)/·\-]+)\s*\(([A-Za-z\s&/\-:]+)\)\s*$'
    m = re.match(subtitle_pattern1, line)
    if m:
        korean_title = m.group(1).strip()
        # 한글 제목에서 괄호 내용 제거
        korean_title = re.sub(r'\([^)]*\)', '', korean_title).strip()
        english_title = m.group(2).strip()
        return {
            "num": korean_title[:20],
            "type": "subsection",
            "title": english_title,
            "level": 2
        }
    
    # 패턴 2: 영문으로 시작 (EDMS, GMP 등)
    subtitle_pattern2 = r'^([A-Z][A-Za-z]*\s+[가-힣][가-힣\s\(\)/·\-]+)\s*\(([A-Za-z\s&/\-:]+)\)\s*$'
    m = re.match(subtitle_pattern2, line)
    if m:
        korean_title = m.group(1).strip()
        korean_title = re.sub(r'\([^)]*\)', '', korean_title).strip()
        english_title = m.group(2).strip()
        return {
            "num": korean_title[:25],
            "type": "subsection",
            "title": english_title,
            "level": 2
        }
    
    return None


def detect_document_format(lines: List[str]) -> str:
    """
    문서 형식 감지
    
    Returns:
        "numbered": 숫자형 (1 목적, 5.1 xxx)
        "named": 이름형 (목적 Purpose, 절차 Procedure)
    """
    numbered_count = 0
    named_count = 0
    
    for line in lines[:50]:  # 첫 50줄만 검사
        line = normalize_text(line.strip())
        if not line:
            continue
        
        # 숫자로 시작하는 섹션
        if re.match(r'^\d+[\.\s]', line):
            numbered_count += 1
        
        # 이름형 섹션
        for keyword in SECTION_KEYWORDS:
            if line.startswith(keyword):
                named_count += 1
                break
    
    return "numbered" if numbered_count > named_count else "named"


# ═══════════════════════════════════════════════════════════════════════════
# 블록 추출 (핵심 로직)
# ═══════════════════════════════════════════════════════════════════════════

def extract_blocks(text: str) -> List[ContentBlock]:
    """
    텍스트에서 섹션 블록 추출
    
    🔥 핵심 로직:
    1. 문서 형식 감지 (숫자형 vs 이름형)
    2. 목차 감지 및 스킵
    3. 섹션 계층 추적 (스택)
    4. section_path 생성
    """
    lines = text.split('\n')
    doc_format = detect_document_format(lines)
    
    blocks = []
    current_lines = []
    current_meta = {"num": None, "type": "intro", "title": "", "level": 0}
    
    # 계층 스택: [{"num": "5", "title": "절차"}, {"num": "5.1", "title": "xxx"}, ...]
    stack = []
    
    # 목차 영역 추적
    in_toc = False
    toc_end_patterns = ['목적', 'Purpose', '1 ', '1.']
    
    # SOP ID 추출
    sop_id = ""
    sop_pattern = re.compile(r'((?:EQ-)?SOP[-_]?\d{4,5})', re.IGNORECASE)
    
    def build_section_path() -> Tuple[str, str]:
        """스택에서 section_path 생성"""
        if not stack:
            return (None, None)
        
        path_parts = [s["num"] for s in stack]
        readable_parts = []
        for s in stack:
            if s["title"]:
                readable_parts.append(f"{s['num']} {s['title']}")
            else:
                readable_parts.append(str(s["num"]))
        
        return (" > ".join(path_parts), " > ".join(readable_parts))
    
    def flush():
        nonlocal current_lines, current_meta
        if current_lines:
            block_text = '\n'.join(current_lines).strip()
            if block_text:
                path, path_readable = build_section_path()
                
                blocks.append(ContentBlock(
                    text=block_text,
                    block_type=current_meta["type"],
                    section=current_meta["num"],
                    metadata={
                        "article_num": current_meta["num"],
                        "article_type": current_meta["type"],
                        "title": current_meta.get("title", ""),
                        "sop_id": sop_id,
                        "section_path": path,
                        "section_path_readable": path_readable,
                    }
                ))
        current_lines = []
    
    for line in lines:
        line_strip = line.strip()
        
        # SOP ID 추출
        sop_match = sop_pattern.search(line_strip)
        if sop_match and not sop_id:
            sop_id = sop_match.group(1).upper().replace('_', '-')
            if not sop_id.startswith('EQ-'):
                sop_id = 'EQ-' + sop_id
        
        # 빈 줄
        if not line_strip:
            current_lines.append(line)
            continue
        
        # 섹션 감지
        section_info = detect_section(line_strip)
        
        # 목차 처리
        if section_info and section_info["type"] == "toc":
            in_toc = True
            flush()
            current_lines = [line]
            current_meta = {"num": "TOC", "type": "toc", "title": "", "level": 0}
            continue
        
        # 목차 종료 감지
        if in_toc:
            for pattern in toc_end_patterns:
                if line_strip.startswith(pattern) and section_info:
                    in_toc = False
                    break
            
            if in_toc:
                current_lines.append(line)
                continue
        
        # 새 섹션 시작
        if section_info:
            flush()
            current_lines = [line]
            
            level = section_info["level"]
            
            # 스택 업데이트
            # 현재 레벨보다 같거나 낮은 항목 제거
            while stack and stack[-1].get("level", 0) >= level:
                stack.pop()
            
            # 현재 섹션 추가
            stack.append({
                "num": section_info["num"],
                "title": section_info["title"],
                "level": level
            })
            
            current_meta = section_info
        else:
            current_lines.append(line)
    
    flush()
    return blocks


# ═══════════════════════════════════════════════════════════════════════════
# DOCX 파싱
# ═══════════════════════════════════════════════════════════════════════════

def load_docx(filename: str, content: bytes) -> ParsedDocument:
    """DOCX 문서 파싱"""
    from docx import Document
    from docx.table import Table
    
    doc = Document(BytesIO(content))
    
    # 텍스트와 테이블 추출 (순서대로)
    full_text_parts = []
    tables_data = []
    
    # 1. 문단 추출
    for para in doc.paragraphs:
        text = para.text.strip()
        if text:
            full_text_parts.append(text)
    
    # 2. 테이블 추출 (텍스트 끝에 추가)
    for table in doc.tables:
        table_text, table_data = _parse_table(table)
        if table_text:
            full_text_parts.append(table_text)
        if table_data:
            tables_data.append(table_data)
    
    full_text = '\n'.join(full_text_parts)
    
    # 블록 추출
    blocks = extract_blocks(full_text)
    
    # 메타데이터
    metadata = {
        "file_name": filename,
        "file_type": ".docx",
        "title": _extract_title(full_text),
        "parser": "docx_v7"
    }
    metadata.update(_extract_sop_metadata(full_text))
    
    return ParsedDocument(
        text=full_text,
        blocks=blocks,
        metadata=metadata,
        tables=tables_data
    )


def _parse_table(table) -> Tuple[str, Dict]:
    """테이블을 텍스트와 구조화 데이터로 변환"""
    rows = []
    for row in table.rows:
        cells = [cell.text.strip() for cell in row.cells]
        rows.append(cells)
    
    if not rows:
        return "", {}
    
    # 마크다운 형식으로 변환
    md_lines = []
    if rows:
        # 헤더
        md_lines.append("| " + " | ".join(rows[0]) + " |")
        md_lines.append("| " + " | ".join(["---"] * len(rows[0])) + " |")
        # 본문
        for row in rows[1:]:
            # 셀 개수 맞추기
            while len(row) < len(rows[0]):
                row.append("")
            md_lines.append("| " + " | ".join(row[:len(rows[0])]) + " |")
    
    table_text = '\n'.join(md_lines)
    table_data = {"rows": rows, "markdown": table_text}
    
    return table_text, table_data


def _extract_title(text: str) -> str:
    """문서 제목 추출"""
    lines = [l.strip() for l in text.split('\n') if l.strip()][:10]
    
    for line in lines:
        # SOP 제목 패턴
        if 'SOP' in line.upper() or '기준서' in line or '규정' in line:
            return line[:100]
    
    return lines[0][:100] if lines else "문서"


def _extract_sop_metadata(text: str) -> Dict:
    """SOP 메타데이터 추출"""
    metadata = {}
    
    # SOP ID
    sop_match = re.search(r'((?:EQ-)?SOP[-_]?\d{4,5})', text, re.IGNORECASE)
    if sop_match:
        sop_id = sop_match.group(1).upper().replace('_', '-')
        if not sop_id.startswith('EQ-'):
            sop_id = 'EQ-' + sop_id
        metadata["sop_id"] = sop_id
    
    # 버전
    ver_match = re.search(r'(?:버전|Version|Rev\.?)\s*[:.]?\s*(\d+(?:\.\d+)?)', text, re.IGNORECASE)
    if ver_match:
        metadata["version"] = ver_match.group(1)
    
    return metadata


# ═══════════════════════════════════════════════════════════════════════════
# 메인 로드 함수
# ═══════════════════════════════════════════════════════════════════════════

def load_document(filename: str, content: bytes) -> ParsedDocument:
    """
    문서 로드 메인 함수
    
    지원 형식: .docx, .doc, .pdf, .txt, .md, .html
    """
    # 확장자 추출 (파일명에 특수 문자가 있을 수 있음)
    filename_lower = filename.lower()
    
    # 실제 확장자 감지
    if '.docx' in filename_lower:
        ext = '.docx'
    elif '.doc' in filename_lower:
        ext = '.doc'
    elif '.pdf' in filename_lower:
        ext = '.pdf'
    elif '.txt' in filename_lower:
        ext = '.txt'
    elif '.md' in filename_lower:
        ext = '.md'
    elif '.html' in filename_lower or '.htm' in filename_lower:
        ext = '.html'
    else:
        ext = Path(filename).suffix.lower()
    
    if ext in [".docx", ".doc"]:
        return load_docx(filename, content)
    
    if ext in [".txt", ".md"]:
        text = content.decode('utf-8', errors='ignore')
        blocks = extract_blocks(text)
        metadata = {
            "file_name": filename,
            "file_type": ext,
            "title": _extract_title(text),
            "parser": "text_v7"
        }
        metadata.update(_extract_sop_metadata(text))
        return ParsedDocument(text=text, blocks=blocks, metadata=metadata)
    
    if ext == ".pdf":
        return _load_pdf(filename, content)
    
    if ext in [".html", ".htm"]:
        return _load_html(filename, content)
    
    # 기본: 텍스트로 처리
    text = content.decode('utf-8', errors='ignore')
    blocks = extract_blocks(text)
    return ParsedDocument(
        text=text,
        blocks=blocks,
        metadata={"file_name": filename, "file_type": ext, "parser": "fallback"}
    )


def _load_pdf(filename: str, content: bytes) -> ParsedDocument:
    """PDF 로드 (Docling 또는 fallback)"""
    try:
        from docling.document_converter import DocumentConverter
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as f:
            f.write(content)
            temp_path = f.name
        
        try:
            converter = DocumentConverter()
            result = converter.convert(temp_path)
            text = result.document.export_to_markdown()
        finally:
            os.unlink(temp_path)
        
        blocks = extract_blocks(text)
        metadata = {
            "file_name": filename,
            "file_type": ".pdf",
            "title": _extract_title(text),
            "parser": "docling"
        }
        metadata.update(_extract_sop_metadata(text))
        return ParsedDocument(text=text, blocks=blocks, metadata=metadata)
        
    except ImportError:
        # PyPDF2 fallback
        try:
            from PyPDF2 import PdfReader
            reader = PdfReader(BytesIO(content))
            text = '\n'.join(page.extract_text() or '' for page in reader.pages)
            blocks = extract_blocks(text)
            return ParsedDocument(
                text=text,
                blocks=blocks,
                metadata={"file_name": filename, "file_type": ".pdf", "parser": "pypdf2"}
            )
        except:
            return ParsedDocument(
                text="[PDF 파싱 실패]",
                blocks=[],
                metadata={"file_name": filename, "file_type": ".pdf", "parser": "failed"}
            )


def _load_html(filename: str, content: bytes) -> ParsedDocument:
    """HTML 로드"""
    from bs4 import BeautifulSoup
    
    html_text = content.decode('utf-8', errors='ignore')
    soup = BeautifulSoup(html_text, 'html.parser')
    
    # 스크립트, 스타일 제거
    for tag in soup(['script', 'style', 'nav', 'footer', 'header']):
        tag.decompose()
    
    text = soup.get_text(separator='\n', strip=True)
    blocks = extract_blocks(text)
    
    title = soup.title.string if soup.title else _extract_title(text)
    
    return ParsedDocument(
        text=text,
        blocks=blocks,
        metadata={"file_name": filename, "file_type": ".html", "title": title, "parser": "bs4"}
    )


# ═══════════════════════════════════════════════════════════════════════════
# 유틸리티
# ═══════════════════════════════════════════════════════════════════════════

def get_supported_extensions() -> List[str]:
    """지원하는 파일 확장자"""
    return [".docx", ".doc", ".pdf", ".txt", ".md", ".html", ".htm"]


# 테스트
if __name__ == "__main__":
    test_text = """
목차 Table of Contents
1 목적 Purpose
2 적용 범위 Scope
3 정의 Definitions
4 책임 Responsibilities
5 절차 Procedure
5.1 품질관리기준서의 구성 및 관리
5.2 시험방법서/규격서/검체채취기준서와의 연계
5.3 시험 수행 및 결과 판정
5.4 일탈(OOS/OOT) 및 조사 시 기준 적용
5.5 개정 및 변경관리
5.6 배포, 교육 및 이행 확인
5.7 기록 및 보관
6 참고문헌 Content Reference
7 첨부 Attachments

1 목적 Purpose
1.1 본 기준서는 품질관리기준서(Quality Control Standard)의 작성, 검토, 승인, 적용 및 개정에 관한 기준을 규정하여, 원자재부터 완제품까지의 품질을 일관되게 확보하고 GMP 요구사항에 적합한 품질관리 체계를 유지하는 것을 목적으로 한다.
The purpose of this standard is to define requirements for preparing, reviewing, approving, implementing, and revising the Quality Control Standard, ensuring consistent quality from raw materials to finished products and maintaining compliance with GMP requirements.

2 적용 범위 Scope
2.1 본 기준서는 회사 내 품질관리(QC) 활동 전반에 적용되며, 원자재, 반제품, 완제품 및 포장자재의 시험, 판정, 방출 및 관련 기록 관리에 적용된다.
This standard applies to overall Quality Control (QC) activities and covers testing, disposition, release, and related record management for raw materials, in-process materials, finished products, and packaging materials.
"""
    
    blocks = extract_blocks(test_text)
    
    print("=" * 60)
    print("블록 추출 결과")
    print("=" * 60)
    
    for i, block in enumerate(blocks):
        print(f"\n[{i}] type={block.metadata.get('article_type')}")
        print(f"    num={block.metadata.get('article_num')}")
        print(f"    path={block.metadata.get('section_path_readable')}")
        print(f"    text={block.text[:50]}...")
