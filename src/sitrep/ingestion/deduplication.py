from __future__ import annotations

from collections import defaultdict

from sitrep.types import DocumentRecord, DuplicateEdge


def build_duplicate_edges(documents: list[DocumentRecord]) -> list[DuplicateEdge]:
    edges: list[DuplicateEdge] = []
    edges.extend(_match_by_key(documents, key_name="canonical_url"))
    edges.extend(_match_by_key(documents, key_name="content_hash"))
    edges.extend(_match_by_title_and_date(documents))
    return _deduplicate_edges(edges)



def select_canonical_documents(
    documents: list[DocumentRecord],
    duplicate_edges: list[DuplicateEdge],
) -> list[DocumentRecord]:
    rejected_ids = {
        edge.right_document_id
        for edge in duplicate_edges
        if edge.preferred_document_id == edge.left_document_id
    }
    return [document for document in documents if document.document_id not in rejected_ids]



def _match_by_key(
    documents: list[DocumentRecord],
    key_name: str,
) -> list[DuplicateEdge]:
    grouped: dict[str, list[DocumentRecord]] = defaultdict(list)
    for document in documents:
        value = getattr(document, key_name)
        if value:
            grouped[str(value)].append(document)

    edges: list[DuplicateEdge] = []
    for matches in grouped.values():
        if len(matches) < 2:
            continue
        preferred = _choose_preferred(matches)
        for other in matches:
            if other.document_id == preferred.document_id:
                continue
            edges.append(
                DuplicateEdge(
                    left_document_id=preferred.document_id,
                    right_document_id=other.document_id,
                    reason=f"exact_{key_name}",
                    score=1.0,
                    preferred_document_id=preferred.document_id,
                )
            )
    return edges



def _match_by_title_and_date(documents: list[DocumentRecord]) -> list[DuplicateEdge]:
    grouped: dict[tuple[str, str], list[DocumentRecord]] = defaultdict(list)
    for document in documents:
        published_day = document.published_at.date().isoformat() if document.published_at else ""
        key = (document.title_hash, published_day)
        if key[0]:
            grouped[key].append(document)

    edges: list[DuplicateEdge] = []
    for matches in grouped.values():
        if len(matches) < 2:
            continue
        preferred = _choose_preferred(matches)
        for other in matches:
            if other.document_id == preferred.document_id:
                continue
            edges.append(
                DuplicateEdge(
                    left_document_id=preferred.document_id,
                    right_document_id=other.document_id,
                    reason="same_title_same_day",
                    score=0.9,
                    preferred_document_id=preferred.document_id,
                )
            )
    return edges



def _choose_preferred(matches: list[DocumentRecord]) -> DocumentRecord:
    return sorted(
        matches,
        key=lambda document: (
            -len(document.clean_text),
            document.source_connector,
            document.document_id,
        ),
    )[0]



def _deduplicate_edges(edges: list[DuplicateEdge]) -> list[DuplicateEdge]:
    unique: dict[tuple[str, str, str], DuplicateEdge] = {}
    for edge in edges:
        unique[(edge.left_document_id, edge.right_document_id, edge.reason)] = edge
    return list(unique.values())
