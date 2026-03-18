from __future__ import annotations

import math
import re
from dataclasses import dataclass

from sitrep.types import ParagraphRecord

TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")


@dataclass(slots=True)
class RetrievalHit:
    paragraph_id: str
    document_id: str
    score: float
    text: str
    title: str
    canonical_url: str
    source_connector: str
    paragraph_index: int
    components: dict[str, float] | None = None



def retrieve_top_k(
    query: str,
    paragraph_records: list[ParagraphRecord],
    *,
    top_k: int = 5,
) -> list[RetrievalHit]:
    hits = score_paragraphs(query, paragraph_records)
    hits.sort(key=lambda hit: (-hit.score, hit.document_id, hit.paragraph_index))
    return hits[:top_k]



def score_paragraphs(
    query: str,
    paragraph_records: list[ParagraphRecord],
) -> list[RetrievalHit]:
    query_tokens = _tokenize(query)
    if not query_tokens:
        return []

    document_frequency: dict[str, int] = {}
    paragraph_tokens: list[tuple[ParagraphRecord, list[str]]] = []
    for paragraph in paragraph_records:
        tokens = _tokenize(paragraph.text)
        paragraph_tokens.append((paragraph, tokens))
        for token in set(tokens):
            document_frequency[token] = document_frequency.get(token, 0) + 1

    total_documents = max(len(paragraph_records), 1)
    hits: list[RetrievalHit] = []
    for paragraph, tokens in paragraph_tokens:
        score = _score(query_tokens, tokens, document_frequency, total_documents)
        if score <= 0:
            continue
        hits.append(
            RetrievalHit(
                paragraph_id=paragraph.paragraph_id,
                document_id=paragraph.document_id,
                score=round(score, 4),
                text=paragraph.text,
                title=paragraph.title,
                canonical_url=paragraph.canonical_url,
                source_connector=paragraph.source_connector,
                paragraph_index=paragraph.paragraph_index,
                components={"lexical": round(score, 4)},
            )
        )
    return hits



def _score(
    query_tokens: list[str],
    paragraph_tokens: list[str],
    document_frequency: dict[str, int],
    total_documents: int,
) -> float:
    if not paragraph_tokens:
        return 0.0
    paragraph_counts: dict[str, int] = {}
    for token in paragraph_tokens:
        paragraph_counts[token] = paragraph_counts.get(token, 0) + 1

    score = 0.0
    paragraph_length = len(paragraph_tokens)
    for token in query_tokens:
        if token not in paragraph_counts:
            continue
        tf = paragraph_counts[token] / paragraph_length
        idf = math.log(1 + total_documents / (1 + document_frequency.get(token, 0)))
        score += tf * idf
    return score



def tokenize(text: str) -> list[str]:
    return _tokenize(text)



def _tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_RE.finditer(text)]
