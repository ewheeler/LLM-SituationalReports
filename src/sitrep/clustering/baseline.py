from __future__ import annotations

import hashlib
import re
from collections import Counter, defaultdict

from sitrep.types import ClusterAssignment, ClusterRecord, ParagraphRecord

TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9-]{2,}")
STOPWORDS = {
    "about", "after", "amid", "among", "and", "are", "been", "being", "but",
    "for", "from", "have", "into", "more", "over", "such", "that", "than",
    "the", "their", "there", "they", "this", "those", "under", "with", "were",
    "will", "would", "response", "situation", "update", "humanitarian",
}


def cluster_paragraphs(paragraph_records: list[ParagraphRecord]) -> tuple[list[ClusterRecord], list[ClusterAssignment]]:
    grouped: dict[str, list[ParagraphRecord]] = defaultdict(list)
    for paragraph in paragraph_records:
        key = _cluster_key(paragraph)
        grouped[key].append(paragraph)

    cluster_records: list[ClusterRecord] = []
    assignments: list[ClusterAssignment] = []
    for group_index, (key, paragraphs) in enumerate(sorted(grouped.items())):
        cluster_id = _cluster_id(key, group_index, paragraphs)
        label, top_terms = _cluster_label(paragraphs)
        document_ids = sorted({paragraph.document_id for paragraph in paragraphs})
        paragraph_ids = [paragraph.paragraph_id for paragraph in paragraphs]
        source_mix = Counter(paragraph.source_connector for paragraph in paragraphs)
        cluster_records.append(
            ClusterRecord(
                cluster_id=cluster_id,
                cluster_label=label,
                event_label=paragraphs[0].event_label if paragraphs else "",
                paragraph_count=len(paragraphs),
                document_count=len(document_ids),
                paragraph_ids=tuple(paragraph_ids),
                document_ids=tuple(document_ids),
                top_terms=tuple(top_terms),
                source_mix=dict(sorted(source_mix.items())),
                representative_paragraph_id=paragraph_ids[0] if paragraph_ids else None,
            )
        )
        for paragraph in paragraphs:
            assignments.append(
                ClusterAssignment(
                    cluster_id=cluster_id,
                    paragraph_id=paragraph.paragraph_id,
                    document_id=paragraph.document_id,
                    paragraph_index=paragraph.paragraph_index,
                    score=1.0,
                )
            )

    cluster_records.sort(key=lambda cluster: (-cluster.paragraph_count, cluster.cluster_label, cluster.cluster_id))
    assignments.sort(key=lambda assignment: (assignment.cluster_id, assignment.document_id, assignment.paragraph_index))
    return cluster_records, assignments



def _cluster_key(paragraph: ParagraphRecord) -> str:
    if paragraph.tags:
        return _normalize_term(paragraph.tags[0])
    tokens = [token for token in _tokenize(paragraph.text) if token not in STOPWORDS]
    if not tokens:
        return "misc"
    top_tokens = [token for token, _ in Counter(tokens).most_common(2)]
    return "::".join(top_tokens)



def _cluster_label(paragraphs: list[ParagraphRecord]) -> tuple[str, list[str]]:
    tokens: list[str] = []
    for paragraph in paragraphs:
        tokens.extend(token for token in _tokenize(paragraph.text) if token not in STOPWORDS)
        tokens.extend(_normalize_term(tag) for tag in paragraph.tags)
    counts = Counter(tokens)
    top_terms = [term for term, _ in counts.most_common(5)]
    if not top_terms:
        return "Miscellaneous Updates", []
    label = " / ".join(term.replace("-", " ").title() for term in top_terms[:2])
    return label, top_terms



def _cluster_id(key: str, group_index: int, paragraphs: list[ParagraphRecord]) -> str:
    seed = f"{group_index}:{key}:{paragraphs[0].event_label if paragraphs else ''}"
    return hashlib.sha1(seed.encode("utf-8")).hexdigest()[:16]



def _tokenize(text: str) -> list[str]:
    return [_normalize_term(match.group(0)) for match in TOKEN_RE.finditer(text)]



def _normalize_term(term: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", term.lower()).strip("-")
