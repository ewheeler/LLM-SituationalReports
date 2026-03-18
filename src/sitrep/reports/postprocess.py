from __future__ import annotations

import copy
import re
from difflib import SequenceMatcher
from typing import Any

CITATION_SEQUENCE_PATTERN = re.compile(r'((?:\[\d+\])+)$')
SDG_NUMBER_PATTERN = re.compile(r'SDG\s+(\d+)')
MIN_SUBSTRING_LENGTH = 50
FUZZY_SIMILARITY_THRESHOLD = 0.7


def finalize_report_payload(
    report_payload: dict[str, object],
    *,
    prefer_section_order: bool = True,
) -> dict[str, object]:
    payload = copy.deepcopy(report_payload)
    grouping = str(payload.get("grouping", "cluster"))
    sections = payload.get("sections", [])
    if not isinstance(sections, list):
        sections = []
        payload["sections"] = sections

    if grouping == "sdg":
        sections.sort(key=_sdg_section_sort_key)

    context_candidates = _normalize_context_candidates(payload.get("context_candidates", []))
    paragraph_candidate_map = _index_candidates_by_paragraph_id(context_candidates)

    paragraph_to_entry: dict[str, dict[str, object]] = {}
    appearance_order: list[str] = []

    def register_citations(citation_entries: list[dict[str, object]]) -> None:
        for entry in _dedup_citation_entries(citation_entries):
            resolved = _resolve_context_entry(entry, context_candidates, paragraph_candidate_map)
            paragraph_id = str(resolved.get("paragraph_id", "")).strip()
            if not paragraph_id:
                continue
            if paragraph_id not in paragraph_to_entry:
                paragraph_to_entry[paragraph_id] = {
                    **resolved,
                    "paragraph_id": paragraph_id,
                }
                appearance_order.append(paragraph_id)

    if prefer_section_order:
        for section in sections:
            register_citations(_as_citation_entries(section.get("citations", [])))
            for answer in _as_answers(section.get("answers", [])):
                register_citations(_as_citation_entries(answer.get("citations", [])))
        executive_summary = _as_dict(payload.get("executive_summary"))
        register_citations(_as_citation_entries(executive_summary.get("citations", [])))
    else:
        executive_summary = _as_dict(payload.get("executive_summary"))
        register_citations(_as_citation_entries(executive_summary.get("citations", [])))
        for section in sections:
            register_citations(_as_citation_entries(section.get("citations", [])))
            for answer in _as_answers(section.get("answers", [])):
                register_citations(_as_citation_entries(answer.get("citations", [])))

    citation_index = {
        paragraph_id: index
        for index, paragraph_id in enumerate(appearance_order, start=1)
    }

    executive_summary = _as_dict(payload.get("executive_summary"))
    executive_summary["citations"] = _reindex_citations(
        _as_citation_entries(executive_summary.get("citations", [])),
        citation_index,
        context_candidates,
        paragraph_candidate_map,
    )
    executive_summary["summary"] = _reattach_citations(executive_summary.get("summary", ""), executive_summary["citations"])
    executive_summary["summary_contexts"] = _context_map(executive_summary["citations"])
    payload["executive_summary"] = executive_summary
    payload["summary_contexts"] = executive_summary["summary_contexts"]

    for section in sections:
        section["citations"] = _reindex_citations(
            _as_citation_entries(section.get("citations", [])),
            citation_index,
            context_candidates,
            paragraph_candidate_map,
        )
        section["summary"] = _reattach_citations(section.get("summary", ""), section["citations"])
        section_contexts = _context_map(section["citations"])
        section["summary_contexts"] = section_contexts
        section["used_contexts"] = section_contexts
        answers = _as_answers(section.get("answers", []))
        for answer in answers:
            answer["citations"] = _reindex_citations(
                _as_citation_entries(answer.get("citations", [])),
                citation_index,
                context_candidates,
                paragraph_candidate_map,
            )
            answer["answer"] = _reattach_citations(answer.get("answer", ""), answer["citations"])
            answer["used_contexts"] = _context_map(answer["citations"])
        section["answers"] = answers

    payload["sections"] = sections
    payload["section_contexts"] = {
        str(section.get("section_id", "")): dict(section.get("summary_contexts", {}))
        for section in sections
        if str(section.get("section_id", ""))
    }
    payload["citations"] = [
        {
            **paragraph_to_entry[paragraph_id],
            "index": citation_index[paragraph_id],
        }
        for paragraph_id in appearance_order
    ]
    payload["context_candidates"] = context_candidates
    payload["postprocessing"] = {
        "citation_order": "sections_then_executive" if prefer_section_order else "executive_then_sections",
        "section_order": "sdg_numeric" if grouping == "sdg" else "original",
        "context_recovery": "exact_then_substring_then_fuzzy",
    }
    return payload


def _sdg_section_sort_key(section: dict[str, Any]) -> tuple[int, str]:
    candidates = [
        str(section.get("title", "")),
        str(section.get("section_id", "")),
    ]
    for candidate in candidates:
        match = SDG_NUMBER_PATTERN.search(candidate)
        if match:
            return (int(match.group(1)), candidate)
        if candidate.lower().startswith("sdg-"):
            try:
                return (int(candidate.split("-", 1)[1]), candidate)
            except ValueError:
                pass
    return (10_000, str(section.get("title", "")))


def _strip_citation_markers(text: str) -> str:
    if not isinstance(text, str):
        return ""
    stripped = CITATION_SEQUENCE_PATTERN.sub("", text).rstrip()
    stripped = re.sub(r'\s+', ' ', stripped)
    return stripped.strip()


def _reattach_citations(text: object, citation_entries: list[dict[str, object]]) -> str:
    base_text = _strip_citation_markers(str(text or ""))
    unique_entries = _dedup_citation_entries(citation_entries)
    if not unique_entries:
        return base_text
    marker = "".join(f"[{int(entry['index'])}]" for entry in unique_entries)
    return f"{base_text} {marker}".strip()


def _reindex_citations(
    citation_entries: list[dict[str, object]],
    citation_index: dict[str, int],
    context_candidates: list[dict[str, object]],
    paragraph_candidate_map: dict[str, dict[str, object]],
) -> list[dict[str, object]]:
    updated: list[dict[str, object]] = []
    for entry in _dedup_citation_entries(citation_entries):
        resolved = _resolve_context_entry(entry, context_candidates, paragraph_candidate_map)
        paragraph_id = str(resolved.get("paragraph_id", "")).strip()
        if not paragraph_id or paragraph_id not in citation_index:
            continue
        updated.append(
            {
                **resolved,
                "paragraph_id": paragraph_id,
                "index": citation_index[paragraph_id],
            }
        )
    return updated


def _resolve_context_entry(
    entry: dict[str, object],
    context_candidates: list[dict[str, object]],
    paragraph_candidate_map: dict[str, dict[str, object]],
) -> dict[str, object]:
    resolved = dict(entry)
    paragraph_id = str(resolved.get("paragraph_id", "")).strip()
    if paragraph_id and paragraph_id in paragraph_candidate_map:
        return _merge_context_fields(resolved, paragraph_candidate_map[paragraph_id])

    if _has_rich_context(resolved):
        return resolved

    context_text = str(resolved.get("context", "") or "").strip()
    if context_text:
        exact = _match_context_exact(context_text, context_candidates)
        if exact is not None:
            return _merge_context_fields(resolved, exact)
        substring = _match_context_substring(context_text, context_candidates)
        if substring is not None:
            return _merge_context_fields(resolved, substring)
        fuzzy = _match_context_fuzzy(context_text, context_candidates)
        if fuzzy is not None:
            return _merge_context_fields(resolved, fuzzy)

    title = str(resolved.get("title", "") or "").strip().lower()
    url = str(resolved.get("url", "") or "").strip()
    for candidate in context_candidates:
        candidate_title = str(candidate.get("title", "") or "").strip().lower()
        candidate_url = str(candidate.get("url", "") or "").strip()
        if title and candidate_title == title:
            return _merge_context_fields(resolved, candidate)
        if url and candidate_url == url:
            return _merge_context_fields(resolved, candidate)
    return resolved


def _merge_context_fields(target: dict[str, object], source: dict[str, object]) -> dict[str, object]:
    merged = dict(target)
    for key in ("paragraph_id", "title", "url", "source_connector", "context", "paragraph_index"):
        if not merged.get(key) and source.get(key):
            merged[key] = source[key]
    return merged


def _has_rich_context(entry: dict[str, object]) -> bool:
    return bool(str(entry.get("context", "") or "").strip())


def _normalize_context_candidates(value: object) -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []
    if not isinstance(value, list):
        return candidates
    for item in value:
        if not isinstance(item, dict):
            continue
        candidates.append(
            {
                "paragraph_id": str(item.get("paragraph_id", "") or "").strip(),
                "title": str(item.get("title", "") or "").strip(),
                "url": str(item.get("url", "") or "").strip(),
                "source_connector": str(item.get("source_connector", "unknown") or "unknown").strip(),
                "context": str(item.get("context", "") or "").strip(),
                "paragraph_index": item.get("paragraph_index"),
            }
        )
    return candidates


def _index_candidates_by_paragraph_id(candidates: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    return {
        candidate["paragraph_id"]: candidate
        for candidate in candidates
        if candidate.get("paragraph_id")
    }


def _match_context_exact(context_text: str, context_candidates: list[dict[str, object]]) -> dict[str, object] | None:
    normalized = _normalize_text(context_text)
    for candidate in context_candidates:
        if _normalize_text(str(candidate.get("context", ""))) == normalized:
            return candidate
    return None


def _match_context_substring(context_text: str, context_candidates: list[dict[str, object]]) -> dict[str, object] | None:
    normalized = _normalize_text(context_text)
    if len(normalized) < MIN_SUBSTRING_LENGTH:
        return None
    for candidate in context_candidates:
        candidate_context = _normalize_text(str(candidate.get("context", "")))
        if normalized and normalized in candidate_context:
            return candidate
        if candidate_context and candidate_context in normalized:
            return candidate
    return None


def _match_context_fuzzy(context_text: str, context_candidates: list[dict[str, object]]) -> dict[str, object] | None:
    normalized = _normalize_text(context_text)
    if not normalized:
        return None
    best_candidate: dict[str, object] | None = None
    best_score = 0.0
    for candidate in context_candidates:
        candidate_context = _normalize_text(str(candidate.get("context", "")))
        if not candidate_context:
            continue
        score = SequenceMatcher(None, normalized, candidate_context).ratio()
        if score > best_score:
            best_score = score
            best_candidate = candidate
    if best_score >= FUZZY_SIMILARITY_THRESHOLD:
        return best_candidate
    return None


def _normalize_text(value: str) -> str:
    return " ".join(value.split()).strip().lower()


def _context_map(citation_entries: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    context_map: dict[str, dict[str, object]] = {}
    for entry in _dedup_citation_entries(citation_entries):
        index = entry.get("index")
        if index is None:
            continue
        context_map[str(index)] = {
            "context": entry.get("context", ""),
            "title": entry.get("title", ""),
            "url": entry.get("url", ""),
            "source_connector": entry.get("source_connector", "unknown"),
            "paragraph_id": entry.get("paragraph_id", ""),
        }
    return context_map


def _dedup_citation_entries(citation_entries: list[dict[str, object]]) -> list[dict[str, object]]:
    seen: set[str] = set()
    unique: list[dict[str, object]] = []
    for entry in citation_entries:
        paragraph_id = str(entry.get("paragraph_id", "")).strip()
        if not paragraph_id or paragraph_id in seen:
            continue
        seen.add(paragraph_id)
        unique.append(entry)
    return unique


def _as_dict(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}


def _as_answers(value: object) -> list[dict[str, object]]:
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _as_citation_entries(value: object) -> list[dict[str, object]]:
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []
