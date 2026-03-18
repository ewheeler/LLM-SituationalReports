from __future__ import annotations

from datetime import datetime, timezone

from sitrep.retrieval.lexical import RetrievalHit, score_paragraphs, tokenize
from sitrep.types import ParagraphRecord

TRUST_PRIORS = {
    "humanitarian_primary": 1.0,
    "major_news": 0.7,
    "curated_web": 0.55,
    "disabled": 0.0,
}


def retrieve_hybrid_top_k(
    query: str,
    paragraph_records: list[ParagraphRecord],
    *,
    top_k: int = 5,
    lexical_weight: float = 1.0,
    title_weight: float = 0.3,
    trust_weight: float = 0.2,
    recency_weight: float = 0.1,
) -> list[RetrievalHit]:
    lexical_hits = score_paragraphs(query, paragraph_records)
    lexical_by_id = {hit.paragraph_id: hit for hit in lexical_hits}
    query_tokens = set(tokenize(query))

    enriched_hits: list[RetrievalHit] = []
    for paragraph in paragraph_records:
        lexical_score = lexical_by_id.get(paragraph.paragraph_id).score if paragraph.paragraph_id in lexical_by_id else 0.0
        title_overlap = _title_overlap(query_tokens, paragraph.title)
        trust_prior = TRUST_PRIORS.get(paragraph.trust_tier, 0.4)
        recency = _recency_score(paragraph.published_at)
        score = (
            lexical_score * lexical_weight
            + title_overlap * title_weight
            + trust_prior * trust_weight
            + recency * recency_weight
        )
        if score <= 0:
            continue
        enriched_hits.append(
            RetrievalHit(
                paragraph_id=paragraph.paragraph_id,
                document_id=paragraph.document_id,
                score=round(score, 4),
                text=paragraph.text,
                title=paragraph.title,
                canonical_url=paragraph.canonical_url,
                source_connector=paragraph.source_connector,
                paragraph_index=paragraph.paragraph_index,
                components={
                    "lexical": round(lexical_score, 4),
                    "title_overlap": round(title_overlap, 4),
                    "trust_prior": round(trust_prior, 4),
                    "recency": round(recency, 4),
                },
            )
        )
    enriched_hits.sort(key=lambda hit: (-hit.score, hit.document_id, hit.paragraph_index))
    return enriched_hits[:top_k]



def _title_overlap(query_tokens: set[str], title: str) -> float:
    if not query_tokens or not title:
        return 0.0
    title_tokens = set(tokenize(title))
    if not title_tokens:
        return 0.0
    return len(query_tokens & title_tokens) / len(query_tokens)



def _recency_score(published_at) -> float:
    if published_at is None:
        return 0.0
    if published_at.tzinfo is None:
        published_at = published_at.replace(tzinfo=timezone.utc)
    age_days = max((datetime.now(timezone.utc) - published_at).days, 0)
    return 1.0 / (1.0 + age_days / 30.0)
