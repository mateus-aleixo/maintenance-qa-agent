"""PDF → cleaned text → overlapping chunks.

Army technical manuals are scanned-era typography with running headers, form feeds
and page furniture. The cleaning here is deliberately conservative: drop obvious
furniture, never rewrite content.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

_WS = re.compile(r"[ \t]+")
_PAGE_NO = re.compile(r"^\s*(?:page\s+)?\d{1,4}\s*$", re.I)
_TM_HEADER = re.compile(r"^\s*TM\s+[\d-]+[A-Z]?\s*$", re.I)


@dataclass(frozen=True)
class Chunk:
    doc: str        # source document name
    page: int       # 1-based page the chunk starts on
    ordinal: int    # position within document
    text: str


def clean_page(raw: str) -> str:
    lines = []
    for line in raw.splitlines():
        line = _WS.sub(" ", line).strip()
        if not line or _PAGE_NO.match(line) or _TM_HEADER.match(line):
            continue
        lines.append(line)
    return "\n".join(lines)


def pdf_pages(path: Path) -> Iterator[tuple[int, str]]:
    # Imported here, not at module scope: store.py imports Chunk from this
    # module, so a top-level pypdf import drags a PDF parser into the serving
    # container, which only ever reads an already-built index.
    from pypdf import PdfReader
    reader = PdfReader(str(path))
    for i, page in enumerate(reader.pages, start=1):
        text = clean_page(page.extract_text() or "")
        if text:
            yield i, text


def chunk_pages(
    doc: str,
    pages: Iterable[tuple[int, str]],
    chunk_chars: int = 1200,
    overlap: int = 200,
) -> Iterator[Chunk]:
    """Greedy character-window chunking over the page stream.

    Chunk boundaries prefer sentence ends; each chunk remembers the page it
    started on so citations point at a real page number.
    """
    if overlap >= chunk_chars:
        raise ValueError("overlap must be smaller than chunk_chars")

    buf = ""
    buf_page = 1
    ordinal = 0
    for page_no, text in pages:
        if not buf:
            buf_page = page_no
        buf = f"{buf}\n{text}" if buf else text
        while len(buf) >= chunk_chars:
            cut = _find_cut(buf, chunk_chars)
            yield Chunk(doc=doc, page=buf_page, ordinal=ordinal, text=buf[:cut].strip())
            ordinal += 1
            buf = buf[max(cut - overlap, 1):]
            buf_page = page_no
    tail = buf.strip()
    if tail:
        yield Chunk(doc=doc, page=buf_page, ordinal=ordinal, text=tail)


def _find_cut(text: str, target: int) -> int:
    """Cut at the last sentence end before `target`, else at the last space.

    A cut is acceptable from a third of the window onward — preferring a real
    sentence boundary over an exact-length chunk."""
    window = text[:target]
    floor = target // 3
    for pat in (". ", ".\n", "? ", "! "):
        pos = window.rfind(pat)
        if pos > floor:
            return pos + 1
    pos = window.rfind(" ")
    return pos if pos > floor else target


def ingest_pdf(path: Path, chunk_chars: int = 1200, overlap: int = 200) -> list[Chunk]:
    return list(chunk_pages(path.name, pdf_pages(path), chunk_chars, overlap))
