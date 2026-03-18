from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from sitrep.sdg.runtime import annotate_question_records_with_sdgs
from sitrep.sdg.taxonomy import SDG_DEFINITIONS
from sitrep.settings import AppSettings
from sitrep.storage.parquet_io import write_records
from sitrep.types import AnswerRecord, QuestionRecord, SdgRecord


_SDG_BY_NUMBER = {definition.number: definition for definition in SDG_DEFINITIONS}


def sdg_question_records(
    app_settings: AppSettings,
    question_records: list[QuestionRecord],
) -> list[QuestionRecord]:
    return annotate_question_records_with_sdgs(app_settings, question_records)


def sdg_records(
    sdg_question_records: list[QuestionRecord],
    answer_records: list[AnswerRecord],
) -> list[SdgRecord]:
    question_ids_by_sdg: dict[int, list[str]] = defaultdict(list)
    answer_ids_by_sdg: dict[int, list[str]] = defaultdict(list)
    event_label = sdg_question_records[0].event_label if sdg_question_records else ""

    for question in sdg_question_records:
        for number in _sdg_numbers(question):
            question_ids_by_sdg[number].append(question.question_id)
            event_label = question.event_label or event_label

    for answer in answer_records:
        for number in _answer_sdg_numbers(answer):
            answer_ids_by_sdg[number].append(answer.answer_id)
            event_label = answer.event_label or event_label

    records: list[SdgRecord] = []
    all_numbers = sorted(set(question_ids_by_sdg) | set(answer_ids_by_sdg))
    for number in all_numbers:
        definition = _SDG_BY_NUMBER.get(number)
        if definition is None:
            continue
        records.append(
            SdgRecord(
                sdg_key=f"sdg-{number}",
                sdg_number=number,
                sdg_label=f"SDG {number} - {definition.name}",
                event_label=event_label,
                question_count=len(question_ids_by_sdg.get(number, [])),
                answer_count=len(answer_ids_by_sdg.get(number, [])),
                question_ids=tuple(question_ids_by_sdg.get(number, [])),
                answer_ids=tuple(answer_ids_by_sdg.get(number, [])),
                metadata={
                    "sdg_name": definition.name,
                    "sdg_description": definition.description,
                },
            )
        )
    return records


def sdg_stats(sdg_question_records: list[QuestionRecord]) -> dict[str, int | float | dict[str, int]]:
    if not sdg_question_records:
        return {
            "sdg_question_records": 0,
            "questions_with_any_sdg": 0,
            "avg_sdg_labels_per_question": 0.0,
            "sdg_label_counts": {},
        }
    label_counts: dict[str, int] = {}
    questions_with_any_sdg = 0
    total_labels = 0
    for question in sdg_question_records:
        labels = question.metadata.get("sdg_labels", [])
        if not isinstance(labels, list):
            labels = []
        if labels:
            questions_with_any_sdg += 1
        total_labels += len(labels)
        for label in labels:
            if isinstance(label, str) and label:
                label_counts[label] = label_counts.get(label, 0) + 1
    return {
        "sdg_question_records": len(sdg_question_records),
        "questions_with_any_sdg": questions_with_any_sdg,
        "avg_sdg_labels_per_question": round(total_labels / len(sdg_question_records), 2),
        "sdg_label_counts": dict(sorted(label_counts.items(), key=lambda item: (-item[1], item[0]))),
    }


def sdg_questions_output_path(
    app_settings: AppSettings,
    event_cache_key: str,
) -> str:
    return str(app_settings.storage.sdg_questions_dir / event_cache_key / "sdg_question_records.parquet")


def sdg_records_output_path(
    app_settings: AppSettings,
    event_cache_key: str,
) -> str:
    return str(app_settings.storage.sdg_questions_dir / event_cache_key / "sdg_records.parquet")


def persisted_sdg_question_records(
    sdg_question_records: list[QuestionRecord],
    sdg_questions_output_path: str,
) -> str:
    return str(write_records(Path(sdg_questions_output_path), sdg_question_records))


def persisted_sdg_records(
    sdg_records: list[SdgRecord],
    sdg_records_output_path: str,
) -> str:
    return str(write_records(Path(sdg_records_output_path), sdg_records))


def persisted_sdg_artifacts(
    persisted_sdg_question_records: str,
    persisted_sdg_records: str,
) -> dict[str, str]:
    return {
        "sdg_question_records": persisted_sdg_question_records,
        "sdg_records": persisted_sdg_records,
    }


def _sdg_numbers(question: QuestionRecord) -> list[int]:
    numbers = question.metadata.get("sdg_numbers", [])
    return [number for number in numbers if isinstance(number, int)] if isinstance(numbers, list) else []


def _answer_sdg_numbers(answer: AnswerRecord) -> list[int]:
    numbers = answer.metadata.get("sdg_numbers", [])
    return [number for number in numbers if isinstance(number, int)] if isinstance(numbers, list) else []
