from __future__ import annotations

import html
import re
from collections import defaultdict

from sitrep.types import AnswerRecord, ClusterRecord, SdgRecord, SummaryRecord


MAX_GROUP_SEGMENTS = 3
MAX_EXECUTIVE_SEGMENTS = 3
MAX_CITATIONS_PER_SEGMENT = 3
MAX_WORDS_PER_SEGMENT = 42
MIN_SEGMENT_CHARS = 40
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def build_cluster_summaries(
    cluster_records: list[ClusterRecord],
    answer_records: list[AnswerRecord],
) -> list[SummaryRecord]:
    answers_by_cluster: dict[str, list[AnswerRecord]] = defaultdict(list)
    for answer in answer_records:
        answers_by_cluster[answer.cluster_id].append(answer)

    summaries: list[SummaryRecord] = []
    for cluster in cluster_records:
        answers = answers_by_cluster.get(cluster.cluster_id, [])
        segments = _segments_from_answers(answers, max_segments=MAX_GROUP_SEGMENTS)
        text = _compose_segment_text(segments)
        citations = _collect_citations(segments)
        summaries.append(
            SummaryRecord(
                summary_id=f"summary-{cluster.cluster_id}",
                event_label=cluster.event_label,
                scope="cluster",
                scope_id=cluster.cluster_id,
                title=cluster.cluster_label,
                summary=text,
                citation_paragraph_ids=citations,
                metadata={
                    "top_terms": list(cluster.top_terms),
                    "paragraph_count": cluster.paragraph_count,
                    "summary_segments": segments,
                },
            )
        )
    return summaries


def build_sdg_summaries(
    sdg_records: list[SdgRecord],
    answer_records: list[AnswerRecord],
) -> list[SummaryRecord]:
    answers_by_sdg: dict[str, list[AnswerRecord]] = defaultdict(list)
    for answer in answer_records:
        for sdg_key in answer.metadata.get("sdg_keys", []):
            if isinstance(sdg_key, str) and sdg_key:
                answers_by_sdg[sdg_key].append(answer)

    summaries: list[SummaryRecord] = []
    for sdg_record in sdg_records:
        answers = answers_by_sdg.get(sdg_record.sdg_key, [])
        segments = _segments_from_answers(answers, max_segments=MAX_GROUP_SEGMENTS)
        text = _compose_segment_text(segments)
        citations = _collect_citations(segments)
        summaries.append(
            SummaryRecord(
                summary_id=f"summary-{sdg_record.sdg_key}",
                event_label=sdg_record.event_label,
                scope="sdg",
                scope_id=sdg_record.sdg_key,
                title=sdg_record.sdg_label,
                summary=text,
                citation_paragraph_ids=citations,
                metadata={
                    "sdg_number": sdg_record.sdg_number,
                    "question_count": sdg_record.question_count,
                    "answer_count": sdg_record.answer_count,
                    "summary_segments": segments,
                },
            )
        )
    return summaries


def build_executive_summary(
    event_label: str,
    summaries: list[SummaryRecord],
    *,
    scope: str = "cluster",
) -> SummaryRecord:
    ranked_summaries = sorted(
        summaries,
        key=lambda summary: (
            len(summary.citation_paragraph_ids),
            len(summary.summary),
        ),
        reverse=True,
    )
    selected = ranked_summaries[:MAX_EXECUTIVE_SEGMENTS]
    segments = _segments_from_summaries(selected)
    summary_text = _compose_segment_text(segments)
    citations = _collect_citations(segments)
    if not summary_text:
        summary_text = f"The available evidence offers only a limited overview of the current {scope} situation."
    return SummaryRecord(
        summary_id=f"executive-{scope}-{event_label.lower().replace(' ', '-')}",
        event_label=event_label,
        scope="executive",
        scope_id=f"{scope}:{event_label}",
        title=f"Executive Summary: {event_label}",
        summary=summary_text,
        citation_paragraph_ids=citations,
        metadata={
            "summary_scope": scope,
            "group_summaries_used": len(selected),
            "summary_segments": segments,
        },
    )


def _segments_from_answers(
    answers: list[AnswerRecord],
    *,
    max_segments: int,
) -> list[dict[str, object]]:
    segments: list[dict[str, object]] = []
    seen_keys: dict[str, int] = {}
    for answer in answers:
        text = _normalize_sentence(answer.answer)
        if not text:
            continue
        citation_ids = tuple(answer.citation_paragraph_ids[:MAX_CITATIONS_PER_SEGMENT])
        segment = {
            "text": text,
            "citation_paragraph_ids": citation_ids,
            "source": "answer",
            "source_id": answer.answer_id,
        }
        _append_or_merge_segment(segments, seen_keys, segment)
        if len(segments) >= max_segments:
            break
    return segments


def _segments_from_summaries(summaries: list[SummaryRecord]) -> list[dict[str, object]]:
    segments: list[dict[str, object]] = []
    seen_keys: dict[str, int] = {}
    for summary in summaries:
        source_segments = summary.metadata.get("summary_segments", [])
        if isinstance(source_segments, list) and source_segments:
            for segment in source_segments:
                if not isinstance(segment, dict):
                    continue
                text = _normalize_sentence(str(segment.get("text", "")))
                citation_ids = tuple(segment.get("citation_paragraph_ids", ()))[:MAX_CITATIONS_PER_SEGMENT]
                if not text:
                    continue
                _append_or_merge_segment(
                    segments,
                    seen_keys,
                    {
                        "text": text,
                        "citation_paragraph_ids": citation_ids,
                        "source": "summary_segment",
                        "source_id": summary.summary_id,
                    },
                )
                if len(segments) >= MAX_EXECUTIVE_SEGMENTS:
                    return segments
            continue
        text = _normalize_sentence(summary.summary)
        citation_ids = tuple(summary.citation_paragraph_ids[:MAX_CITATIONS_PER_SEGMENT])
        if not text:
            continue
        _append_or_merge_segment(
            segments,
            seen_keys,
            {
                "text": text,
                "citation_paragraph_ids": citation_ids,
                "source": "summary",
                "source_id": summary.summary_id,
            },
        )
        if len(segments) >= MAX_EXECUTIVE_SEGMENTS:
            break
    return segments


def _compose_segment_text(segments: list[dict[str, object]]) -> str:
    if not segments:
        return ""
    return " ".join(str(segment.get("text", "")).strip() for segment in segments if str(segment.get("text", "")).strip())


def _append_or_merge_segment(
    segments: list[dict[str, object]],
    seen_keys: dict[str, int],
    segment: dict[str, object],
) -> None:
    key = _segment_key(str(segment.get("text", "")))
    if not key:
        return
    existing_index = seen_keys.get(key)
    if existing_index is None:
        seen_keys[key] = len(segments)
        segments.append(segment)
        return
    existing = segments[existing_index]
    existing_ids = list(existing.get("citation_paragraph_ids", ()))
    for paragraph_id in segment.get("citation_paragraph_ids", ()): 
        if isinstance(paragraph_id, str) and paragraph_id not in existing_ids:
            existing_ids.append(paragraph_id)
    existing["citation_paragraph_ids"] = tuple(existing_ids[:MAX_CITATIONS_PER_SEGMENT])



def _segment_key(text: str) -> str:
    normalized = re.sub(r"[^a-z0-9 ]+", " ", text.lower())
    return " ".join(normalized.split())


def _collect_citations(segments: list[dict[str, object]]) -> tuple[str, ...]:
    ordered: list[str] = []
    for segment in segments:
        for paragraph_id in segment.get("citation_paragraph_ids", ()): 
            if isinstance(paragraph_id, str) and paragraph_id and paragraph_id not in ordered:
                ordered.append(paragraph_id)
    return tuple(ordered)


def _normalize_sentence(text: str) -> str:
    normalized = html.unescape(" ".join((text or "").split()).strip())
    if not normalized:
        return ""

    lower = normalized.lower()
    content_marker = lower.find(" content: ")
    if content_marker != -1:
        normalized = normalized[content_marker + len(" content: ") :].strip()
    else:
        normalized = re.sub(r"\bOriginal URL:\s*\S+", "", normalized, flags=re.IGNORECASE)
        normalized = re.sub(r"\btitle:\s*", "", normalized, flags=re.IGNORECASE)
        normalized = re.sub(r"\bcontent:\s*", "", normalized, flags=re.IGNORECASE)

    normalized = normalized.replace("&quot;", '"').replace("&#x27;", "'")
    sentences = [sentence.strip(" \"'") for sentence in SENTENCE_SPLIT_RE.split(normalized) if sentence.strip()]

    candidate = ""
    for sentence in sentences:
        cleaned = _trim_words(sentence)
        if len(cleaned) >= MIN_SEGMENT_CHARS:
            candidate = cleaned
            break
    if not candidate and sentences:
        candidate = _trim_words(sentences[0])
    if not candidate:
        candidate = _trim_words(normalized)
    if candidate and candidate[-1] not in ".!?":
        candidate = candidate + "."
    return candidate


def _trim_words(text: str) -> str:
    words = text.split()
    if len(words) <= MAX_WORDS_PER_SEGMENT:
        return " ".join(words).strip()
    return " ".join(words[:MAX_WORDS_PER_SEGMENT]).strip() + " …"
