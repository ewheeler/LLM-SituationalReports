from __future__ import annotations

from pathlib import Path

from sitrep.llm.runtime import generate_question_records
from sitrep.settings import AppSettings
from sitrep.storage.parquet_io import write_records
from sitrep.types import ClusterRecord, ParagraphRecord, QuestionRecord


def question_records(
    app_settings: AppSettings,
    cluster_records: list[ClusterRecord],
    clustered_paragraph_records: list[ParagraphRecord],
) -> list[QuestionRecord]:
    return generate_question_records(app_settings, cluster_records, clustered_paragraph_records)


def question_stats(question_records: list[QuestionRecord]) -> dict[str, int | float]:
    if not question_records:
        return {
            "question_records": 0,
            "avg_questions_per_cluster": 0.0,
        }
    cluster_ids = {question.cluster_id for question in question_records}
    avg_per_cluster = len(question_records) / max(len(cluster_ids), 1)
    return {
        "question_records": len(question_records),
        "avg_questions_per_cluster": round(avg_per_cluster, 2),
    }


def questions_output_path(
    app_settings: AppSettings,
    event_cache_key: str,
) -> str:
    return str(app_settings.storage.questions_dir / event_cache_key / "question_records.parquet")


def persisted_question_records(
    question_records: list[QuestionRecord],
    questions_output_path: str,
) -> str:
    return str(write_records(Path(questions_output_path), question_records))


def persisted_question_artifacts(
    persisted_question_records: str,
) -> dict[str, str]:
    return {
        "question_records": persisted_question_records,
    }
