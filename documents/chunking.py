"""Split parsed sections into overlapping chunks for embedding.

Strategy (a simple "recursive" splitter):
1. Split text into pieces at the coarsest boundary that keeps each piece under the
   chunk size: paragraphs, then lines, then sentences, then words.
2. Greedily pack consecutive pieces into chunks of up to `chunk_size` tokens.
3. Start each next chunk with the last ~`overlap` tokens of the previous one, so an idea
   cut by a boundary is still complete in at least one chunk.

Boundaries:
- PDF pages are a layout artefact — sentences continue on the next page — so chunks may
  span pages and record the page range they cover.
- DOCX headings are semantic boundaries: a new heading always starts a new chunk (with
  no overlap), so one vector never mixes two topics.

Token counts are *estimates* (see `estimate_tokens`); exact usage comes back from the
embedding API.
"""

import math
import re
from dataclasses import dataclass

from .parsing import Section

# Rough average for modern tokenizers on mixed English/Russian text. Cyrillic needs more
# tokens per character than Latin, so this errs towards smaller chunks — the safe side.
CHARS_PER_TOKEN = 3.5

_SEPARATORS = [
    re.compile(r"\n\n+"),  # paragraphs
    re.compile(r"\n"),  # lines
    re.compile(r"(?<=[.!?…])\s+"),  # sentences
    re.compile(r"\s+"),  # words
]


def estimate_tokens(text: str) -> int:
    return math.ceil(len(text) / CHARS_PER_TOKEN)


@dataclass(frozen=True)
class ChunkData:
    text: str
    token_count: int
    page_start: int | None = None
    page_end: int | None = None
    section: str = ""


@dataclass
class _Piece:
    text: str
    page: int | None
    heading: str


def chunk_sections(
    sections: list[Section], chunk_size: int = 500, overlap: int = 75
) -> list[ChunkData]:
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")

    pieces = [
        _Piece(text, section.page, section.heading)
        for section in sections
        for text in _split(section.text, chunk_size)
    ]

    chunks: list[ChunkData] = []
    current: list[_Piece] = []
    for piece in pieces:
        if current and piece.heading != current[-1].heading:
            chunks.append(_make_chunk(current))
            current = []
        elif current and _tokens([*current, piece]) > chunk_size:
            chunks.append(_make_chunk(current))
            # Shrink the overlap if the tail plus the new piece would not fit.
            budget = min(overlap, chunk_size - estimate_tokens(piece.text) - 1)
            current = _overlap_tail(current, budget)
        current.append(piece)
    if current:
        chunks.append(_make_chunk(current))
    return chunks


def _tokens(pieces: list[_Piece]) -> int:
    return estimate_tokens("\n\n".join(p.text for p in pieces))


def _split(text: str, chunk_size: int, level: int = 0) -> list[str]:
    """Recursively split text until every piece fits into chunk_size tokens."""
    text = text.strip()
    if not text:
        return []
    if estimate_tokens(text) <= chunk_size:
        return [text]
    if level >= len(_SEPARATORS):
        # A single "word" longer than a chunk (e.g. a base64 blob): hard cut.
        step = int(chunk_size * CHARS_PER_TOKEN)
        return [text[i : i + step] for i in range(0, len(text), step)]
    parts = [p for p in _SEPARATORS[level].split(text) if p.strip()]
    if len(parts) == 1:
        return _split(text, chunk_size, level + 1)
    return [piece for part in parts for piece in _split(part, chunk_size, level + 1)]


def _overlap_tail(pieces: list[_Piece], budget: int) -> list[_Piece]:
    """The end of a chunk, up to `budget` tokens, to repeat at the start of the next one.

    Whole pieces are taken while they fit; the piece that does not fit contributes its
    trailing sentences (or words). Without this, overlap would only ever happen when a
    whole paragraph is shorter than the overlap — i.e. almost never on real documents.
    """
    tail: list[_Piece] = []
    for piece in reversed(pieces):
        if _tokens([piece, *tail]) <= budget:
            tail.insert(0, piece)
            continue
        remaining = budget - (_tokens(tail) + 1 if tail else 0)
        partial = _tail_text(piece.text, remaining)
        if partial:
            tail.insert(0, _Piece(partial, piece.page, piece.heading))
        break
    return tail


def _tail_text(text: str, budget: int) -> str:
    """Trailing sentences of `text` within budget; trailing words if no sentence fits."""
    for separator in (_SEPARATORS[2], _SEPARATORS[3]):  # sentences, then words
        parts = [p for p in separator.split(text) if p.strip()]
        taken: list[str] = []
        for part in reversed(parts):
            if estimate_tokens(" ".join([part, *taken])) > budget:
                break
            taken.insert(0, part)
        if taken:
            return " ".join(taken)
    return ""


def _make_chunk(pieces: list[_Piece]) -> ChunkData:
    text = "\n\n".join(p.text for p in pieces)
    pages = [p.page for p in pieces if p.page is not None]
    headings = [p.heading for p in pieces if p.heading]
    return ChunkData(
        text=text,
        token_count=estimate_tokens(text),
        page_start=min(pages) if pages else None,
        page_end=max(pages) if pages else None,
        section=headings[0] if headings else "",
    )
