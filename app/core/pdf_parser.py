"""
app/core/pdf_parser.py — Extract plain text from PDF files, page by page.

Uses PyMuPDF (fitz) — fastest Python PDF library, handles scanned PDFs
better than pdfplumber and doesn't require Java like Tika.
"""

from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF


@dataclass
class PageContent:
    page_number: int   # 1-indexed
    text: str
    char_count: int


@dataclass
class ParsedDocument:
    filename: str
    total_pages: int
    pages: list[PageContent]

    @property
    def full_text(self) -> str:
        return "\n\n".join(p.text for p in self.pages if p.text.strip())

    @property
    def total_chars(self) -> int:
        return sum(p.char_count for p in self.pages)


def parse_pdf(file_path: str | Path) -> ParsedDocument:
    """
    Open a PDF and extract text content per page.

    Args:
        file_path: Path to the PDF file.

    Returns:
        ParsedDocument with per-page text and metadata.

    Raises:
        FileNotFoundError: If the path doesn't exist.
        ValueError: If the file is not a readable PDF.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    try:
        doc = fitz.open(str(path))
    except Exception as exc:
        raise ValueError(f"Cannot open PDF '{path.name}': {exc}") from exc

    pages: list[PageContent] = []

    for page_idx in range(len(doc)):
        page = doc[page_idx]
        # get_text("text") returns plain text; "blocks" gives layout info
        raw_text: str = page.get_text("text")
        # Normalise whitespace without destroying paragraph breaks
        cleaned = _clean_text(raw_text)
        pages.append(
            PageContent(
                page_number=page_idx + 1,
                text=cleaned,
                char_count=len(cleaned),
            )
        )

    doc.close()

    return ParsedDocument(
        filename=path.name,
        total_pages=len(pages),
        pages=pages,
    )


def parse_pdf_bytes(data: bytes, filename: str = "upload.pdf") -> ParsedDocument:
    """
    Parse a PDF from raw bytes (e.g. an uploaded file in memory).
    """
    try:
        doc = fitz.open(stream=data, filetype="pdf")
    except Exception as exc:
        raise ValueError(f"Cannot parse PDF bytes: {exc}") from exc

    pages: list[PageContent] = []

    for page_idx in range(len(doc)):
        page = doc[page_idx]
        raw_text: str = page.get_text("text")
        cleaned = _clean_text(raw_text)
        pages.append(
            PageContent(
                page_number=page_idx + 1,
                text=cleaned,
                char_count=len(cleaned),
            )
        )

    doc.close()

    return ParsedDocument(
        filename=filename,
        total_pages=len(pages),
        pages=pages,
    )


def _clean_text(text: str) -> str:
    """
    Light normalisation:
    - Strip leading/trailing whitespace per line
    - Collapse runs of 3+ blank lines to 2
    - Remove null bytes and control characters (except newlines/tabs)
    """
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        lines.append(stripped)

    # Collapse excessive blank lines
    result_lines: list[str] = []
    blank_count = 0
    for line in lines:
        if line == "":
            blank_count += 1
            if blank_count <= 2:
                result_lines.append("")
        else:
            blank_count = 0
            result_lines.append(line)

    return "\n".join(result_lines).strip()
