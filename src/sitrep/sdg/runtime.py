from __future__ import annotations

from dataclasses import replace

from sitrep.llm.openai_client import OpenAIResponsesClient
from sitrep.llm.registry import get_prompt_spec
from sitrep.llm.runtime import BASELINE_PROVIDER, OPENAI_PROVIDER
from sitrep.settings import AppSettings, GenerationStageSettings
from sitrep.sdg.baseline import BaselineSdgClassifier
from sitrep.sdg.taxonomy import SDG_DEFINITIONS, sdg_descriptions_for_prompt
from sitrep.types import QuestionRecord


class SdgProviderError(RuntimeError):
    pass


def annotate_question_records_with_sdgs(
    app_settings: AppSettings,
    question_records: list[QuestionRecord],
) -> list[QuestionRecord]:
    provider = app_settings.sdg.provider
    if provider == BASELINE_PROVIDER:
        classifier = BaselineSdgClassifier(app_settings.sdg)
        return [
            _annotated_question_record(
                question_record,
                classifier.classify(question_record.question, _question_context(question_record)),
                provider,
            )
            for question_record in question_records
        ]
    if provider != OPENAI_PROVIDER:
        raise SdgProviderError(f"Unsupported SDG provider: {provider}")
    if app_settings.run.offline_mode:
        raise SdgProviderError(
            "SDG provider 'openai' is unavailable while offline_mode is enabled. Use baseline SDG classification or disable offline mode."
        )
    client = OpenAIResponsesClient.from_settings(app_settings)
    prompt_id = app_settings.sdg.prompt_id or "sdg.classify.v1"
    prompt_spec = get_prompt_spec(prompt_id)
    stage_settings = GenerationStageSettings(
        provider=provider,
        model=app_settings.sdg.model,
        prompt_id=prompt_id,
        temperature=0.0,
        max_output_tokens=500,
    )
    sdg_descriptions = sdg_descriptions_for_prompt()
    annotated: list[QuestionRecord] = []
    for question_record in question_records:
        rendered_prompt = prompt_spec.prompt.render(
            question=question_record.question,
            sdg_descriptions=sdg_descriptions,
            cluster_label=question_record.metadata.get("cluster_label"),
            top_terms=question_record.metadata.get("top_terms", []),
        )
        payload = client.generate_structured(
            prompt_spec=prompt_spec,
            stage_settings=stage_settings,
            rendered_prompt=rendered_prompt,
        )
        scores = payload.get("scores", [])
        if not isinstance(scores, list) or len(scores) != len(SDG_DEFINITIONS):
            raise SdgProviderError(
                f"SDG classifier returned {len(scores) if isinstance(scores, list) else 'invalid'} scores; expected {len(SDG_DEFINITIONS)}."
            )
        normalized_scores = [1 if int(score) else 0 for score in scores]
        labels = [
            definition.name
            for definition, score in zip(SDG_DEFINITIONS, normalized_scores)
            if score
        ][: app_settings.sdg.max_labels_per_question]
        numbers = [
            definition.number
            for definition, score in zip(SDG_DEFINITIONS, normalized_scores)
            if score
        ][: app_settings.sdg.max_labels_per_question]
        metadata = {
            "sdg_scores": normalized_scores,
            "sdg_labels": labels,
            "sdg_numbers": numbers,
            "sdg_rationale": payload.get("rationale", ""),
        }
        annotated.append(_annotated_question_record(question_record, metadata, provider))
    return annotated


def _question_context(question_record: QuestionRecord) -> str:
    cluster_label = question_record.metadata.get("cluster_label")
    top_terms = question_record.metadata.get("top_terms", [])
    parts: list[str] = []
    if isinstance(cluster_label, str) and cluster_label.strip():
        parts.append(cluster_label)
    if isinstance(top_terms, list):
        parts.extend(str(term) for term in top_terms if isinstance(term, str) and term.strip())
    return " ".join(parts).strip()


def _annotated_question_record(
    question_record: QuestionRecord,
    sdg_payload: dict[str, object],
    provider: str,
) -> QuestionRecord:
    metadata = dict(question_record.metadata)
    metadata.update(sdg_payload)
    metadata["sdg_provider"] = provider
    return replace(question_record, metadata=metadata)
