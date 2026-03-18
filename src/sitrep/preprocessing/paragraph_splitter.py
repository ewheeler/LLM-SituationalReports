from __future__ import annotations

import hashlib
import re
from dataclasses import replace

from sitrep.preprocessing.normalize import normalize_document_text, normalize_paragraph_text
from sitrep.types import DocumentRecord, ParagraphRecord


DEFAULT_MAX_PARAGRAPH_CHARS = 1200
DEFAULT_MIN_PARAGRAPH_CHARS = 80
SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+")


def split_document_into_paragraphs(
    document: DocumentRecord,
    *,
    max_chars: int = DEFAULT_MAX_PARAGRAPH_CHARS,
    min_chars: int = DEFAULT_MIN_PARAGRAPH_CHARS,
) -> list[ParagraphRecord]:
    normalized_text = normalize_document_text(document.clean_text or document.raw_text)
    if not normalized_text:
        return []

    raw_segments = _split_into_segments(normalized_text)
    merged_segments = _merge_short_segments(raw_segments, min_chars=min_chars)
    chunked_segments = _chunk_long_segments(merged_segments, max_chars=max_chars)

    paragraphs: list[ParagraphRecord] = []
    cursor = 0
    for index, segment in enumerate(chunked_segments):
        text = normalize_paragraph_text(segment)
        if not text:
            continue
        start_char = normalized_text.find(segment, cursor)
        if start_char < 0:
            start_char = cursor
        end_char = start_char + len(segment)
        cursor = end_char
        paragraphs.append(
            ParagraphRecord(
                paragraph_id=_paragraph_id(document.document_id, index, text),
                document_id=document.document_id,
                paragraph_index=index,
                text=text,
                start_char=start_char,
                end_char=end_char,
                source_connector=document.source_connector,
                publisher=document.publisher,
                canonical_url=document.canonical_url,
                retrieved_url=document.retrieved_url,
                title=document.title,
                published_at=document.published_at,
                language=document.language,
                country_codes=document.country_codes,
                event_label=document.event_label,
                tags=document.tags,
                trust_tier=document.trust_tier,
                attachments=document.attachments,
                metadata={
                    "content_hash": document.content_hash,
                    "title_hash": document.title_hash,
                },
            )
        )
    return paragraphs


def _split_into_segments(text: str) -> list[str]:
    blocks = [block.strip() for block in text.split("\n\n") if block.strip()]
    if blocks:
        return blocks
    return [text]


def _merge_short_segments(segments: list[str], *, min_chars: int) -> list[str]:
    if not segments:
        return []
    merged: list[str] = []
    buffer = ""
    for segment in segments:
        candidate = f"{buffer} {segment}".strip() if buffer else segment
        if len(candidate) < min_chars:
            buffer = candidate
            continue
        if buffer and candidate != segment:
            merged.append(candidate)
            buffer = ""
            continue
        if buffer:
            merged.append(buffer)
            buffer = ""
        merged.append(segment)
    if buffer:
        if merged:
            merged[-1] = f"{merged[-1]} {buffer}".strip()
        else:
            merged.append(buffer)
    return merged


def _chunk_long_segments(segments: list[str], *, max_chars: int) -> list[str]:
    chunks: list[str] = []
    for segment in segments:
        if len(segment) <= max_chars:
            chunks.append(segment)
            continue
        sentences = [sentence.strip() for sentence in SENTENCE_BOUNDARY_RE.split(segment) if sentence.strip()]
        current = ""
        for sentence in sentences:
            candidate = f"{current} {sentence}".strip() if current else sentence
            if len(candidate) <= max_chars:
                current = candidate
                continue
            if current:
                chunks.append(current)
            if len(sentence) <= max_chars:
                current = sentence
                continue
            chunks.extend(_hard_wrap(sentence, max_chars=max_chars))
            current = ""
        if current:
            chunks.append(current)
    return chunks


def _hard_wrap(text: str, *, max_chars: int) -> list[str]:
    words = text.split()
    if not words:
        return []
    chunks: list[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if len(candidate) <= max_chars:
            current = candidate
            continue
        chunks.append(current)
        current = word
    if current:
        chunks.append(current)
    return chunks


def _paragraph_id(document_id: str, index: int, text: str) -> str:
    digest = hashlib.sha1(f"{document_id}:{index}:{text}".encode("utf-8")).hexdigest()[:16]
    return f"{document_id[:12]}-{index:04d}-{digest}"
