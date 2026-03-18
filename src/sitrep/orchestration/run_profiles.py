from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RunProfile:
    name: str
    module_paths: tuple[str, ...]
    outputs: tuple[str, ...]


INGEST_ONLY = RunProfile(
    name="ingest_only",
    module_paths=(
        "sitrep.ingestion.nodes",
        "sitrep.preprocessing.nodes",
        "sitrep.clustering.nodes",
        "sitrep.questions.nodes",
        "sitrep.sdg.nodes",
        "sitrep.answers.nodes",
        "sitrep.summaries.nodes",
        "sitrep.reports.nodes",
    ),
    outputs=(
        "documents_manifest",
        "duplicate_edges",
        "source_mix",
        "ingestion_stats",
        "paragraph_stats",
        "cluster_stats",
        "question_stats",
        "sdg_stats",
        "answer_stats",
        "summary_stats",
        "persisted_ingestion_artifacts",
        "persisted_preprocessing_artifacts",
        "persisted_clustering_artifacts",
        "persisted_question_artifacts",
        "persisted_sdg_artifacts",
        "persisted_answer_artifacts",
        "persisted_summary_artifacts",
        "persisted_report_artifacts",
        "persisted_run_manifest",
    ),
)


DEFAULT_RUN_PROFILES: dict[str, RunProfile] = {
    INGEST_ONLY.name: INGEST_ONLY,
}


def get_run_profile(name: str = "ingest_only") -> RunProfile:
    try:
        return DEFAULT_RUN_PROFILES[name]
    except KeyError as error:
        available = ", ".join(sorted(DEFAULT_RUN_PROFILES))
        raise KeyError(f"Unknown run profile '{name}'. Available profiles: {available}") from error
