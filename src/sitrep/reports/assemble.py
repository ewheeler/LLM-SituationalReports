from __future__ import annotations

from collections import defaultdict

from sitrep.types import AnswerRecord, SummaryRecord


def build_report_payload(
    event_name: str,
    executive_summary: SummaryRecord,
    section_summaries: list[SummaryRecord],
    answer_records: list[AnswerRecord],
    *,
    grouping: str = "cluster",
) -> dict[str, object]:
    answers_by_scope = _group_answers(answer_records, grouping)
    citation_registry = _build_citation_registry(section_summaries, answer_records, executive_summary)
    context_candidates = _build_context_candidates(answer_records, citation_registry)
    sections = []
    for summary in section_summaries:
        answers = answers_by_scope.get(summary.scope_id, [])
        summary_citations = [_citation_entry(citation_registry, paragraph_id) for paragraph_id in summary.citation_paragraph_ids]
        section_answers = []
        for answer in answers:
            answer_citations = [_citation_entry(citation_registry, paragraph_id) for paragraph_id in answer.citation_paragraph_ids]
            section_answers.append(
                {
                    "question_id": answer.question_id,
                    "question": answer.metadata.get("question", answer.question_id),
                    "question_type": answer.metadata.get("question_type", ""),
                    "answer_id": answer.answer_id,
                    "answer": _attach_citation_markers(answer.answer, answer.citation_paragraph_ids, citation_registry),
                    "citations": answer_citations,
                    "used_contexts": _context_map(answer_citations),
                }
            )
        section_contexts = _context_map(summary_citations)
        sections.append(
            {
                "section_id": summary.scope_id,
                "section_scope": summary.scope,
                "title": summary.title,
                "summary": _render_summary_text(summary, citation_registry),
                "citations": summary_citations,
                "summary_contexts": section_contexts,
                "used_contexts": section_contexts,
                "answers": section_answers,
            }
        )

    executive_citations = [_citation_entry(citation_registry, paragraph_id) for paragraph_id in executive_summary.citation_paragraph_ids]
    executive_contexts = _context_map(executive_citations)
    return {
        "event_name": event_name,
        "grouping": grouping,
        "executive_summary": {
            "title": executive_summary.title,
            "summary": _render_summary_text(executive_summary, citation_registry),
            "citations": executive_citations,
            "summary_contexts": executive_contexts,
        },
        "summary_contexts": executive_contexts,
        "sections": sections,
        "section_contexts": {
            section["section_id"]: section["summary_contexts"]
            for section in sections
        },
        "citations": [
            citation_registry[paragraph_id]
            for paragraph_id in sorted(citation_registry, key=lambda pid: citation_registry[pid]["index"])
        ],
        "context_candidates": context_candidates,
    }


def build_markdown_report(report_payload: dict[str, object]) -> str:
    executive = report_payload["executive_summary"]
    sections = report_payload["sections"]
    citations = report_payload["citations"]
    grouping = str(report_payload.get("grouping", "cluster")).replace("_", " ").title()
    lines = [f"# {report_payload['event_name']}", "", f"_Report grouping: {grouping}_", "", "## Executive Summary", ""]
    lines.append(executive["summary"])
    lines.append("")
    for section in sections:
        lines.extend([
            f"## {section['title']}",
            "",
            section["summary"],
            "",
        ])
        for answer in section["answers"]:
            heading = answer.get("question") or answer["question_id"]
            lines.extend([
                f"### {heading}",
                "",
                answer["answer"],
                "",
            ])
    if citations:
        lines.extend(["## Citations", ""])
        for citation in citations:
            lines.append(f"[{citation['index']}] {citation['title']} — {citation['source_connector']} — {citation['url']}")
    return "\n".join(lines).strip() + "\n"



def _render_summary_text(summary: SummaryRecord, registry: dict[str, dict[str, object]]) -> str:
    segments = summary.metadata.get("summary_segments", [])
    if isinstance(segments, list) and segments:
        rendered_segments: list[str] = []
        for segment in segments:
            if not isinstance(segment, dict):
                continue
            text = str(segment.get("text", "")).strip()
            paragraph_ids = tuple(segment.get("citation_paragraph_ids", ()))
            if not text:
                continue
            rendered_segments.append(_attach_citation_markers(text, paragraph_ids, registry))
        if rendered_segments:
            return " ".join(rendered_segments)
    return _attach_citation_markers(summary.summary, summary.citation_paragraph_ids, registry)


def _group_answers(answer_records: list[AnswerRecord], grouping: str) -> dict[str, list[AnswerRecord]]:
    grouped: dict[str, list[AnswerRecord]] = defaultdict(list)
    if grouping == "sdg":
        for answer in answer_records:
            for sdg_key in answer.metadata.get("sdg_keys", []):
                if isinstance(sdg_key, str) and sdg_key:
                    grouped[sdg_key].append(answer)
        return grouped
    for answer in answer_records:
        grouped[answer.cluster_id].append(answer)
    return grouped


def _build_citation_registry(
    section_summaries: list[SummaryRecord],
    answer_records: list[AnswerRecord],
    executive_summary: SummaryRecord,
) -> dict[str, dict[str, object]]:
    registry: dict[str, dict[str, object]] = {}
    ordered_ids: list[str] = []
    detail_sources = []
    for answer in answer_records:
        detail_sources.extend(answer.metadata.get("citation_details", []))
    detail_by_id = {
        detail["paragraph_id"]: detail
        for detail in detail_sources
        if isinstance(detail, dict) and detail.get("paragraph_id")
    }
    all_ids = [
        *executive_summary.citation_paragraph_ids,
        *(paragraph_id for summary in section_summaries for paragraph_id in summary.citation_paragraph_ids),
        *(paragraph_id for answer in answer_records for paragraph_id in answer.citation_paragraph_ids),
    ]
    for paragraph_id in all_ids:
        if paragraph_id not in ordered_ids:
            ordered_ids.append(paragraph_id)
    for index, paragraph_id in enumerate(ordered_ids, start=1):
        detail = detail_by_id.get(paragraph_id, {})
        registry[paragraph_id] = {
            "index": index,
            "paragraph_id": paragraph_id,
            "title": detail.get("title", paragraph_id),
            "url": detail.get("url", ""),
            "source_connector": detail.get("source_connector", "unknown"),
            "context": detail.get("context", ""),
            "paragraph_index": detail.get("paragraph_index"),
        }
    return registry


def _build_context_candidates(
    answer_records: list[AnswerRecord],
    citation_registry: dict[str, dict[str, object]],
) -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()
    for entry in citation_registry.values():
        candidate = _candidate_from_entry(entry)
        key = (str(candidate.get("paragraph_id", "")), str(candidate.get("context", "")))
        if key not in seen:
            seen.add(key)
            candidates.append(candidate)
    for answer in answer_records:
        for collection_name in ("citation_details", "retrieval_hits"):
            for detail in answer.metadata.get(collection_name, []):
                if not isinstance(detail, dict):
                    continue
                candidate = {
                    "paragraph_id": detail.get("paragraph_id", ""),
                    "title": detail.get("title", "") or detail.get("document_title", ""),
                    "url": detail.get("url", "") or detail.get("canonical_url", ""),
                    "source_connector": detail.get("source_connector", "unknown"),
                    "context": detail.get("context", "") or detail.get("text", "") or detail.get("text_preview", ""),
                    "paragraph_index": detail.get("paragraph_index"),
                }
                key = (str(candidate.get("paragraph_id", "")), str(candidate.get("context", "")))
                if key not in seen:
                    seen.add(key)
                    candidates.append(candidate)
    return candidates


def _candidate_from_entry(entry: dict[str, object]) -> dict[str, object]:
    return {
        "paragraph_id": entry.get("paragraph_id", ""),
        "title": entry.get("title", ""),
        "url": entry.get("url", ""),
        "source_connector": entry.get("source_connector", "unknown"),
        "context": entry.get("context", ""),
        "paragraph_index": entry.get("paragraph_index"),
    }


def _attach_citation_markers(text: str, paragraph_ids: tuple[str, ...], registry: dict[str, dict[str, object]]) -> str:
    if not text or not paragraph_ids:
        return text
    marker = " " + "".join(f"[{registry[paragraph_id]['index']}]" for paragraph_id in paragraph_ids if paragraph_id in registry)
    return (text + marker).strip()


def _context_map(citation_entries: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    context_map: dict[str, dict[str, object]] = {}
    for entry in citation_entries:
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


def _citation_entry(registry: dict[str, dict[str, object]], paragraph_id: str) -> dict[str, object]:
    return registry.get(paragraph_id, {
        "index": 0,
        "paragraph_id": paragraph_id,
        "title": paragraph_id,
        "url": "",
        "source_connector": "unknown",
        "context": "",
        "paragraph_index": None,
    })
