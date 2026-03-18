from __future__ import annotations

from pathlib import Path

from sitrep.llm.runtime import generate_answer_records
from sitrep.settings import AppSettings
from sitrep.storage.parquet_io import write_records
from sitrep.types import AnswerRecord, ParagraphRecord, QuestionRecord


def answer_records(
    app_settings: AppSettings,
    sdg_question_records: list[QuestionRecord],
    clustered_paragraph_records: list[ParagraphRecord],
) -> list[AnswerRecord]:
    return generate_answer_records(
        app_settings,
        sdg_question_records,
        clustered_paragraph_records,
    )


def answer_stats(answer_records: list[AnswerRecord]) -> dict[str, int | float]:
    if not answer_records:
        return {
            "answer_records": 0,
            "answers_with_citations": 0,
            "avg_citations_per_answer": 0.0,
        }
    answers_with_citations = sum(1 for answer in answer_records if answer.citation_paragraph_ids)
    avg_citations = sum(len(answer.citation_paragraph_ids) for answer in answer_records) / len(answer_records)
    return {
        "answer_records": len(answer_records),
        "answers_with_citations": answers_with_citations,
        "avg_citations_per_answer": round(avg_citations, 2),
    }


def answers_output_path(
    app_settings: AppSettings,
    event_cache_key: str,
) -> str:
    return str(app_settings.storage.answers_dir / event_cache_key / "answer_records.parquet")


def persisted_answer_records(
    answer_records: list[AnswerRecord],
    answers_output_path: str,
) -> str:
    return str(write_records(Path(answers_output_path), answer_records))


def persisted_answer_artifacts(
    persisted_answer_records: str,
) -> dict[str, str]:
    return {
        "answer_records": persisted_answer_records,
    }
