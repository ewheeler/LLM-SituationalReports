from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, replace

from sitrep.llm.openai_client import OpenAIResponsesClient
from sitrep.llm.registry import get_prompt_spec
from sitrep.questions.notebook_style import generate_questions_openai_notebook_style
from sitrep.retrieval.hybrid import retrieve_hybrid_top_k
from sitrep.settings import AppSettings
from sitrep.types import AnswerRecord, ClusterRecord, ParagraphRecord, QuestionRecord, SdgRecord, SummaryRecord


class GenerationProviderError(RuntimeError):
    pass


BASELINE_PROVIDER = "baseline"
OPENAI_PROVIDER = "openai"



def generate_question_records(
    app_settings: AppSettings,
    cluster_records: list[ClusterRecord],
    clustered_paragraph_records: list[ParagraphRecord],
) -> list[QuestionRecord]:
    stage = app_settings.generation.questions
    if stage.provider == BASELINE_PROVIDER:
        from sitrep.questions.baseline import generate_questions

        return generate_questions(cluster_records)
    if stage.provider != OPENAI_PROVIDER:
        raise GenerationProviderError(f"Unsupported question provider: {stage.provider}")
    _ensure_online_generation(app_settings, stage.provider)
    return generate_questions_openai_notebook_style(
        app_settings,
        cluster_records,
        clustered_paragraph_records,
    )



def _group_paragraphs_by_cluster(paragraph_records: list[ParagraphRecord]) -> dict[str, list[ParagraphRecord]]:
    grouped: dict[str, list[ParagraphRecord]] = {}
    for paragraph in paragraph_records:
        cluster_id = paragraph.metadata.get("cluster_id")
        if not isinstance(cluster_id, str) or not cluster_id:
            continue
        grouped.setdefault(cluster_id, []).append(paragraph)
    return grouped



def generate_answer_records(
    app_settings: AppSettings,
    question_records: list[QuestionRecord],
    paragraph_records: list[ParagraphRecord],
) -> list[AnswerRecord]:
    stage = app_settings.generation.answers
    if stage.provider == BASELINE_PROVIDER:
        from sitrep.answers.baseline import answer_questions

        retrieval = app_settings.retrieval
        return answer_questions(
            question_records,
            paragraph_records,
            top_k=retrieval.top_k,
            lexical_weight=retrieval.lexical_weight,
            title_weight=retrieval.title_weight,
            trust_weight=retrieval.trust_weight,
            recency_weight=retrieval.recency_weight,
        )
    if stage.provider != OPENAI_PROVIDER:
        raise GenerationProviderError(f"Unsupported answer provider: {stage.provider}")
    _ensure_online_generation(app_settings, stage.provider)
    client = OpenAIResponsesClient.from_settings(app_settings)
    prompt_id = stage.prompt_id or "answers.grounded.v1"
    prompt_spec = get_prompt_spec(prompt_id)
    retrieval = app_settings.retrieval
    paragraphs_by_cluster = _group_paragraphs_by_cluster(paragraph_records)
    answers: list[AnswerRecord] = []
    for question in question_records:
        candidate_paragraphs = paragraphs_by_cluster.get(question.cluster_id, paragraph_records)
        hits = retrieve_hybrid_top_k(
            question.question,
            candidate_paragraphs,
            top_k=retrieval.top_k,
            lexical_weight=retrieval.lexical_weight,
            title_weight=retrieval.title_weight,
            trust_weight=retrieval.trust_weight,
            recency_weight=retrieval.recency_weight,
        )
        rendered_prompt = prompt_spec.prompt.render(
            question=asdict(question),
            cluster={"cluster_id": question.cluster_id, "event_label": question.event_label},
            evidence=[
                {
                    "paragraph_id": hit.paragraph_id,
                    "title": hit.title,
                    "source_connector": hit.source_connector,
                    "canonical_url": hit.canonical_url,
                    "score": hit.score,
                    "text": hit.text,
                    "components": hit.components or {},
                }
                for hit in hits
            ],
        )
        effective_stage = _scaled_summary_stage_settings(stage, rendered_prompt)
        payload = client.generate_structured(
            prompt_spec=prompt_spec,
            stage_settings=effective_stage,
            rendered_prompt=rendered_prompt,
        )
        allowed_ids = {hit.paragraph_id for hit in hits}
        citation_ids = tuple(
            paragraph_id
            for paragraph_id in payload.get("citation_paragraph_ids", [])
            if isinstance(paragraph_id, str) and paragraph_id in allowed_ids
        )
        answers.append(
            AnswerRecord(
                answer_id=f"{question.question_id}-answer",
                question_id=question.question_id,
                cluster_id=question.cluster_id,
                event_label=question.event_label,
                answer=str(payload.get("answer", "")).strip(),
                citation_paragraph_ids=citation_ids,
                confidence=float(payload.get("confidence", 0.0)),
                metadata={
                    "question": question.question,
                    "question_type": question.question_type,
                    "priority": question.priority,
                    "cluster_label": question.metadata.get("cluster_label"),
                    "top_terms": question.metadata.get("top_terms", []),
                    "sdg_numbers": question.metadata.get("sdg_numbers", []),
                    "sdg_labels": question.metadata.get("sdg_labels", []),
                    "sdg_scores": question.metadata.get("sdg_scores", []),
                    "sdg_keys": [f"sdg-{number}" for number in question.metadata.get("sdg_numbers", []) if isinstance(number, int)],
                    "retrieval_method": "hybrid_openai",
                    "generation_provider": stage.provider,
                    "generation_model": stage.model or prompt_spec.default_model,
                    "generation_prompt_id": prompt_spec.prompt.prompt_id,
                    "generation_max_output_tokens": effective_stage.max_output_tokens,
                    "generation_rationale": payload.get("rationale", ""),
                    "retrieval_hits": [
                        {
                            "paragraph_id": hit.paragraph_id,
                            "score": hit.score,
                            "title": hit.title,
                            "canonical_url": hit.canonical_url,
                            "source_connector": hit.source_connector,
                            "paragraph_index": hit.paragraph_index,
                            "components": hit.components or {},
                            "text_preview": hit.text[:240],
                        }
                        for hit in hits
                    ],
                    "citation_details": [
                        {
                            "paragraph_id": hit.paragraph_id,
                            "title": hit.title,
                            "url": hit.canonical_url,
                            "source_connector": hit.source_connector,
                            "context": hit.text,
                            "paragraph_index": hit.paragraph_index,
                        }
                        for hit in hits
                    ],
                },
            )
        )
    return answers



def generate_sdg_summaries(
    app_settings: AppSettings,
    sdg_records: list[SdgRecord],
    answer_records: list[AnswerRecord],
) -> list[SummaryRecord]:
    stage = app_settings.generation.cluster_summaries
    if stage.provider == BASELINE_PROVIDER:
        from sitrep.summaries.baseline import build_sdg_summaries

        return build_sdg_summaries(sdg_records, answer_records)
    if stage.provider != OPENAI_PROVIDER:
        raise GenerationProviderError(f"Unsupported SDG summary provider: {stage.provider}")
    _ensure_online_generation(app_settings, stage.provider)
    client = OpenAIResponsesClient.from_settings(app_settings)
    prompt_id = stage.prompt_id or "summaries.cluster.v1"
    prompt_spec = get_prompt_spec(prompt_id)
    answers_by_sdg: dict[str, list[AnswerRecord]] = defaultdict(list)
    for answer in answer_records:
        for sdg_key in answer.metadata.get("sdg_keys", []):
            if isinstance(sdg_key, str) and sdg_key:
                answers_by_sdg[sdg_key].append(answer)

    summaries: list[SummaryRecord] = []
    for sdg_record in sdg_records:
        answers = answers_by_sdg.get(sdg_record.sdg_key, [])
        rendered_prompt = prompt_spec.prompt.render(
            cluster=asdict(sdg_record),
            answers=[asdict(answer) for answer in answers],
        )
        effective_stage = _scaled_summary_stage_settings(stage, rendered_prompt)
        payload = client.generate_structured(
            prompt_spec=prompt_spec,
            stage_settings=effective_stage,
            rendered_prompt=rendered_prompt,
        )
        allowed_ids = {
            paragraph_id
            for answer in answers
            for paragraph_id in answer.citation_paragraph_ids
        }
        summary_segments = _normalize_summary_segments(payload, allowed_ids)
        citation_ids = _summary_citation_ids(payload, allowed_ids, summary_segments)
        summary_text = _summary_text(payload, summary_segments)
        summaries.append(
            SummaryRecord(
                summary_id=f"summary-{sdg_record.sdg_key}",
                event_label=sdg_record.event_label,
                scope="sdg",
                scope_id=sdg_record.sdg_key,
                title=sdg_record.sdg_label,
                summary=summary_text,
                citation_paragraph_ids=citation_ids,
                metadata={
                    "sdg_number": sdg_record.sdg_number,
                    "answer_count": sdg_record.answer_count,
                    "question_count": sdg_record.question_count,
                    "summary_segments": summary_segments,
                    "generation_provider": stage.provider,
                    "generation_model": stage.model or prompt_spec.default_model,
                    "generation_prompt_id": prompt_spec.prompt.prompt_id,
                    "generation_max_output_tokens": effective_stage.max_output_tokens,
                    "generation_rationale": payload.get("rationale", ""),
                },
            )
        )
    return summaries


def generate_cluster_summaries(
    app_settings: AppSettings,
    cluster_records: list[ClusterRecord],
    answer_records: list[AnswerRecord],
) -> list[SummaryRecord]:
    stage = app_settings.generation.cluster_summaries
    if stage.provider == BASELINE_PROVIDER:
        from sitrep.summaries.baseline import build_cluster_summaries

        return build_cluster_summaries(cluster_records, answer_records)
    if stage.provider != OPENAI_PROVIDER:
        raise GenerationProviderError(f"Unsupported cluster summary provider: {stage.provider}")
    _ensure_online_generation(app_settings, stage.provider)
    client = OpenAIResponsesClient.from_settings(app_settings)
    prompt_id = stage.prompt_id or "summaries.cluster.v1"
    prompt_spec = get_prompt_spec(prompt_id)
    answers_by_cluster: dict[str, list[AnswerRecord]] = defaultdict(list)
    for answer in answer_records:
        answers_by_cluster[answer.cluster_id].append(answer)

    summaries: list[SummaryRecord] = []
    for cluster in cluster_records:
        answers = answers_by_cluster.get(cluster.cluster_id, [])
        rendered_prompt = prompt_spec.prompt.render(
            cluster=asdict(cluster),
            answers=[asdict(answer) for answer in answers],
        )
        effective_stage = _scaled_summary_stage_settings(stage, rendered_prompt)
        payload = client.generate_structured(
            prompt_spec=prompt_spec,
            stage_settings=effective_stage,
            rendered_prompt=rendered_prompt,
        )
        allowed_ids = {
            paragraph_id
            for answer in answers
            for paragraph_id in answer.citation_paragraph_ids
        }
        summary_segments = _normalize_summary_segments(payload, allowed_ids)
        citation_ids = _summary_citation_ids(payload, allowed_ids, summary_segments)
        summary_text = _summary_text(payload, summary_segments)
        summaries.append(
            SummaryRecord(
                summary_id=f"summary-{cluster.cluster_id}",
                event_label=cluster.event_label,
                scope="cluster",
                scope_id=cluster.cluster_id,
                title=str(payload.get("title", cluster.cluster_label)).strip() or cluster.cluster_label,
                summary=summary_text,
                citation_paragraph_ids=citation_ids,
                metadata={
                    "top_terms": list(cluster.top_terms),
                    "paragraph_count": cluster.paragraph_count,
                    "summary_segments": summary_segments,
                    "generation_provider": stage.provider,
                    "generation_model": stage.model or prompt_spec.default_model,
                    "generation_prompt_id": prompt_spec.prompt.prompt_id,
                    "generation_max_output_tokens": effective_stage.max_output_tokens,
                    "generation_rationale": payload.get("rationale", ""),
                },
            )
        )
    return summaries



def generate_executive_summary(
    app_settings: AppSettings,
    event_name: str,
    cluster_summaries: list[SummaryRecord],
) -> SummaryRecord:
    stage = app_settings.generation.executive_summary
    if stage.provider == BASELINE_PROVIDER:
        from sitrep.summaries.baseline import build_executive_summary

        return build_executive_summary(event_name, cluster_summaries)
    if stage.provider != OPENAI_PROVIDER:
        raise GenerationProviderError(f"Unsupported executive summary provider: {stage.provider}")
    _ensure_online_generation(app_settings, stage.provider)
    client = OpenAIResponsesClient.from_settings(app_settings)
    prompt_id = stage.prompt_id or "summaries.executive.v1"
    prompt_spec = get_prompt_spec(prompt_id)
    grounded_summary_count = sum(
        1 for summary in cluster_summaries if summary.summary.strip() and summary.citation_paragraph_ids
    )
    unique_citation_count = len(
        {
            paragraph_id
            for summary in cluster_summaries
            for paragraph_id in summary.citation_paragraph_ids
        }
    )
    rendered_prompt = prompt_spec.prompt.render(
        event_name=event_name,
        grounded_summary_count=grounded_summary_count,
        unique_citation_count=unique_citation_count,
        cluster_summaries=[asdict(summary) for summary in cluster_summaries],
    )
    effective_stage = _scaled_summary_stage_settings(stage, rendered_prompt)
    payload = client.generate_structured(
        prompt_spec=prompt_spec,
        stage_settings=effective_stage,
        rendered_prompt=rendered_prompt,
    )
    allowed_ids = {
        paragraph_id
        for summary in cluster_summaries
        for paragraph_id in summary.citation_paragraph_ids
    }
    summary_segments = _normalize_summary_segments(payload, allowed_ids)
    citation_ids = _summary_citation_ids(payload, allowed_ids, summary_segments)
    summary_text = _summary_text(payload, summary_segments)
    return SummaryRecord(
        summary_id=f"executive-{event_name.lower().replace(' ', '-')}",
        event_label=event_name,
        scope="executive",
        scope_id=event_name,
        title=str(payload.get("title", f"Executive Summary: {event_name}")).strip() or f"Executive Summary: {event_name}",
        summary=summary_text,
        citation_paragraph_ids=citation_ids,
        metadata={
            "cluster_summaries_used": len(cluster_summaries),
            "summary_segments": summary_segments,
            "generation_provider": stage.provider,
            "generation_model": stage.model or prompt_spec.default_model,
            "generation_prompt_id": prompt_spec.prompt.prompt_id,
            "generation_max_output_tokens": effective_stage.max_output_tokens,
            "generation_rationale": payload.get("rationale", ""),
        },
    )



def _scaled_summary_stage_settings(stage, rendered_prompt):
    if not getattr(stage, "scale_output_with_input", False):
        return stage
    prompt_text = f"{rendered_prompt.system_prompt}\n{rendered_prompt.user_prompt}".strip()
    estimated_input_tokens = _estimate_text_tokens(prompt_text)
    scaled_budget = int(
        stage.max_output_tokens
        + (estimated_input_tokens * stage.output_token_estimate_ratio)
        + stage.output_token_buffer
    )
    if stage.max_output_tokens_cap is not None:
        scaled_budget = min(scaled_budget, stage.max_output_tokens_cap)
    scaled_budget = max(stage.max_output_tokens, scaled_budget)
    return replace(stage, max_output_tokens=scaled_budget)



def _estimate_text_tokens(text: str) -> int:
    normalized = " ".join((text or "").split())
    if not normalized:
        return 0
    return max(1, len(normalized) // 4)



def _normalize_summary_segments(
    payload: dict[str, object],
    allowed_ids: set[str],
) -> list[dict[str, object]]:
    segments: list[dict[str, object]] = []
    for item in payload.get("summary_segments", []):
        if not isinstance(item, dict):
            continue
        text = " ".join(str(item.get("text", "")).split()).strip()
        citation_ids = _filter_citation_ids(item.get("citation_paragraph_ids", []), allowed_ids, max_items=4)
        if not text or not citation_ids:
            continue
        segments.append(
            {
                "text": text,
                "citation_paragraph_ids": citation_ids,
            }
        )
    if segments:
        return segments

    fallback_text = " ".join(str(payload.get("summary", "")).split()).strip()
    fallback_ids = _filter_citation_ids(payload.get("citation_paragraph_ids", []), allowed_ids, max_items=12)
    return _fallback_summary_segments(fallback_text, fallback_ids)



def _summary_citation_ids(
    payload: dict[str, object],
    allowed_ids: set[str],
    summary_segments: list[dict[str, object]],
) -> tuple[str, ...]:
    ordered: list[str] = []
    for segment in summary_segments:
        for paragraph_id in segment.get("citation_paragraph_ids", ()):  # type: ignore[union-attr]
            if isinstance(paragraph_id, str) and paragraph_id and paragraph_id not in ordered:
                ordered.append(paragraph_id)
    if ordered:
        return tuple(ordered)
    return _filter_citation_ids(payload.get("citation_paragraph_ids", []), allowed_ids, max_items=12)



def _summary_text(
    payload: dict[str, object],
    summary_segments: list[dict[str, object]],
) -> str:
    segment_text = " ".join(
        str(segment.get("text", "")).strip()
        for segment in summary_segments
        if str(segment.get("text", "")).strip()
    ).strip()
    if segment_text:
        return segment_text
    return " ".join(str(payload.get("summary", "")).split()).strip()



def _filter_citation_ids(
    values: object,
    allowed_ids: set[str],
    *,
    max_items: int,
) -> tuple[str, ...]:
    ordered: list[str] = []
    if not isinstance(values, list):
        return tuple()
    for paragraph_id in values:
        if not isinstance(paragraph_id, str) or paragraph_id not in allowed_ids or paragraph_id in ordered:
            continue
        ordered.append(paragraph_id)
        if len(ordered) >= max_items:
            break
    return tuple(ordered)



def _fallback_summary_segments(
    summary_text: str,
    citation_ids: tuple[str, ...],
) -> list[dict[str, object]]:
    if not summary_text:
        return []
    fragments = [fragment.strip() for fragment in summary_text.replace("\n", " ").split(". ") if fragment.strip()]
    if not fragments:
        fragments = [summary_text.strip()]
    segments: list[dict[str, object]] = []
    chunk_size = max(1, len(citation_ids) // max(1, min(len(fragments), 4)))
    for index, fragment in enumerate(fragments[:4]):
        text = fragment if fragment.endswith((".", "!", "?")) else f"{fragment}."
        start = min(index * chunk_size, len(citation_ids))
        end = min(len(citation_ids), start + max(1, chunk_size))
        segment_ids = citation_ids[start:end] or citation_ids[: min(2, len(citation_ids))]
        if not segment_ids:
            continue
        segments.append({"text": text, "citation_paragraph_ids": segment_ids})
    return segments



def _ensure_online_generation(app_settings: AppSettings, provider: str) -> None:
    if app_settings.run.offline_mode:
        raise GenerationProviderError(
            f"Provider '{provider}' is unavailable while offline_mode is enabled. Use baseline generation or disable offline mode."
        )
