from __future__ import annotations

import re
from dataclasses import asdict
from difflib import SequenceMatcher
from typing import Any

from sitrep.llm.openai_client import OpenAIResponsesClient
from sitrep.llm.registry import get_prompt_spec
from sitrep.settings import AppSettings, GenerationStageSettings
from sitrep.types import ClusterRecord, ParagraphRecord, QuestionRecord

DEFAULT_PROMPT_IDS = (
    "questions.cluster.v1",
    "questions.cluster.v2",
    "questions.cluster.v3",
)
QUESTION_TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9-]{2,}")


class QuestionDeduper:
    def __init__(self) -> None:
        self._cross_encoder = self._load_cross_encoder()

    def similarity(self, left: str, right: str) -> float:
        if self._cross_encoder is not None:
            try:
                score = float(self._cross_encoder.predict([(left, right)])[0])
                return max(0.0, min(1.0, score))
            except Exception:
                pass
        return _lexical_similarity(left, right)

    @staticmethod
    def _load_cross_encoder():
        try:
            from sentence_transformers import CrossEncoder  # type: ignore
        except ModuleNotFoundError:
            return None
        try:
            return CrossEncoder("cross-encoder/quora-roberta-base", device="cpu")
        except Exception:
            return None



def generate_questions_openai_notebook_style(
    app_settings: AppSettings,
    cluster_records: list[ClusterRecord],
    clustered_paragraph_records: list[ParagraphRecord],
) -> list[QuestionRecord]:
    client = OpenAIResponsesClient.from_settings(app_settings)
    stage = app_settings.generation.questions
    prompt_ids = _question_prompt_ids(stage)
    paragraphs_by_cluster = _group_paragraphs_by_cluster(clustered_paragraph_records)
    deduper = QuestionDeduper()

    question_records: list[QuestionRecord] = []
    selected_global_questions: list[str] = []
    for cluster_index, cluster in enumerate(cluster_records):
        evidence = _cluster_evidence(paragraphs_by_cluster.get(cluster.cluster_id, []))
        candidates: list[dict[str, Any]] = []
        for prompt_id in prompt_ids:
            prompt_spec = get_prompt_spec(prompt_id)
            rendered_prompt = prompt_spec.prompt.render(
                event_name=cluster.event_label,
                cluster=asdict(cluster),
                evidence=evidence,
            )
            payload = client.generate_structured(
                prompt_spec=prompt_spec,
                stage_settings=stage,
                rendered_prompt=rendered_prompt,
            )
            for item in payload.get("questions", []):
                question = _normalize_question(str(item.get("question", "")).strip())
                if not question or not question.endswith("?"):
                    continue
                candidates.append(
                    {
                        "question": question,
                        "question_type": _question_type(question, str(item.get("question_type", "analysis"))),
                        "priority": int(item.get("priority", max(1, 100 - cluster_index))),
                        "prompt_id": prompt_id,
                        "rationale": str(item.get("rationale", "")),
                    }
                )
        selected = _select_unique_questions(
            candidates,
            deduper,
            selected_global_questions,
            within_cluster_threshold=stage.within_cluster_dedupe_threshold,
            across_cluster_threshold=stage.across_cluster_dedupe_threshold,
            max_questions=stage.max_questions_per_cluster,
        )
        for question_index, item in enumerate(selected, start=1):
            question_records.append(
                QuestionRecord(
                    question_id=f"{cluster.cluster_id}-{question_index:02d}",
                    cluster_id=cluster.cluster_id,
                    event_label=cluster.event_label,
                    question=item["question"],
                    question_type=item["question_type"],
                    priority=int(item["priority"]),
                    metadata={
                        "cluster_label": cluster.cluster_label,
                        "top_terms": list(cluster.top_terms),
                        "generation_provider": stage.provider,
                        "generation_model": stage.model or "gpt-5.4-mini",
                        "generation_prompt_id": item["prompt_id"],
                        "generation_rationale": item["rationale"],
                        "candidate_prompt_ids": list(prompt_ids),
                        "evidence_paragraph_ids": [item_["paragraph_id"] for item_ in evidence],
                        "within_cluster_dedupe_threshold": stage.within_cluster_dedupe_threshold,
                        "across_cluster_dedupe_threshold": stage.across_cluster_dedupe_threshold,
                        "notebook_style": True,
                    },
                )
            )
            selected_global_questions.append(item["question"])
    return question_records



def _question_prompt_ids(stage: GenerationStageSettings) -> tuple[str, ...]:
    if stage.fanout_prompt_ids:
        return tuple(stage.fanout_prompt_ids)
    preferred_prompt_id = stage.prompt_id
    if preferred_prompt_id and preferred_prompt_id in DEFAULT_PROMPT_IDS:
        ordered = [preferred_prompt_id, *(prompt_id for prompt_id in DEFAULT_PROMPT_IDS if prompt_id != preferred_prompt_id)]
        return tuple(ordered)
    return DEFAULT_PROMPT_IDS



def _group_paragraphs_by_cluster(
    clustered_paragraph_records: list[ParagraphRecord],
) -> dict[str, list[ParagraphRecord]]:
    grouped: dict[str, list[ParagraphRecord]] = {}
    for paragraph in clustered_paragraph_records:
        cluster_id = paragraph.metadata.get("cluster_id")
        if not isinstance(cluster_id, str) or not cluster_id:
            continue
        grouped.setdefault(cluster_id, []).append(paragraph)
    return grouped



def _cluster_evidence(paragraphs: list[ParagraphRecord]) -> list[dict[str, Any]]:
    ranked = sorted(
        paragraphs,
        key=lambda paragraph: (
            -float(paragraph.metadata.get("assignment_score", 0.0)),
            paragraph.title,
            paragraph.paragraph_index,
        ),
    )
    evidence = []
    for paragraph in ranked[:8]:
        evidence.append(
            {
                "paragraph_id": paragraph.paragraph_id,
                "title": paragraph.title,
                "source_connector": paragraph.source_connector,
                "canonical_url": paragraph.canonical_url,
                "published_at": paragraph.published_at.isoformat() if paragraph.published_at else None,
                "text": paragraph.text,
            }
        )
    return evidence



def _select_unique_questions(
    candidates: list[dict[str, Any]],
    deduper: QuestionDeduper,
    selected_global_questions: list[str],
    *,
    within_cluster_threshold: float,
    across_cluster_threshold: float,
    max_questions: int,
) -> list[dict[str, Any]]:
    if not candidates:
        return []
    selected: list[dict[str, Any]] = []
    for candidate in candidates:
        question = candidate["question"]
        if any(deduper.similarity(existing["question"], question) >= within_cluster_threshold for existing in selected):
            continue
        if any(deduper.similarity(existing_question, question) >= across_cluster_threshold for existing_question in selected_global_questions):
            continue
        selected.append(candidate)
        if len(selected) >= max_questions:
            break
    return selected



def _normalize_question(question: str) -> str:
    question = re.sub(r"^\d+[\.)]\s*", "", question).strip()
    question = re.sub(r"\s+", " ", question)
    if question and not question.endswith("?") and any(word in question.lower() for word in ("what", "which", "how", "who", "where", "when", "why")):
        question += "?"
    return question



def _question_type(question: str, suggested: str) -> str:
    normalized = question.lower()
    if "who" in normalized or "affected" in normalized:
        return "affected_groups"
    if any(term in normalized for term in ("where", "which areas", "which locations", "geographic")):
        return "geography"
    if any(term in normalized for term in ("gap", "constraint", "response", "measure", "action")):
        return "actions_and_gaps"
    if any(term in normalized for term in ("risk", "threat", "hazard", "forecast")):
        return "risks"
    if suggested and suggested != "analysis":
        return suggested
    return "developments"



def _lexical_similarity(left: str, right: str) -> float:
    left_tokens = {match.group(0).lower() for match in QUESTION_TOKEN_RE.finditer(left)}
    right_tokens = {match.group(0).lower() for match in QUESTION_TOKEN_RE.finditer(right)}
    if not left_tokens or not right_tokens:
        return SequenceMatcher(None, left, right).ratio()
    jaccard = len(left_tokens & right_tokens) / max(len(left_tokens | right_tokens), 1)
    ratio = SequenceMatcher(None, left.lower(), right.lower()).ratio()
    return max(jaccard, ratio)
