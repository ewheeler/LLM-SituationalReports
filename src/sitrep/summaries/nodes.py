from __future__ import annotations

from pathlib import Path

from sitrep.llm.runtime import generate_cluster_summaries, generate_executive_summary, generate_sdg_summaries
from sitrep.settings import AppSettings
from sitrep.storage.parquet_io import write_records
from sitrep.types import AnswerRecord, ClusterRecord, SdgRecord, SummaryRecord



def cluster_summaries(
    app_settings: AppSettings,
    cluster_records: list[ClusterRecord],
    answer_records: list[AnswerRecord],
) -> list[SummaryRecord]:
    return generate_cluster_summaries(app_settings, cluster_records, answer_records)



def sdg_summaries(
    app_settings: AppSettings,
    sdg_records: list[SdgRecord],
    answer_records: list[AnswerRecord],
) -> list[SummaryRecord]:
    return generate_sdg_summaries(app_settings, sdg_records, answer_records)



def executive_summary(
    app_settings: AppSettings,
    event_name: str,
    cluster_summaries: list[SummaryRecord],
) -> SummaryRecord:
    return generate_executive_summary(app_settings, event_name, cluster_summaries)



def sdg_executive_summary(
    app_settings: AppSettings,
    event_name: str,
    sdg_summaries: list[SummaryRecord],
) -> SummaryRecord:
    return generate_executive_summary(app_settings, event_name, sdg_summaries)



def summary_stats(
    cluster_summaries: list[SummaryRecord],
    sdg_summaries: list[SummaryRecord],
) -> dict[str, int | float]:
    return {
        "cluster_summaries": len(cluster_summaries),
        "sdg_summaries": len(sdg_summaries),
    }



def summaries_output_path(
    app_settings: AppSettings,
    event_cache_key: str,
) -> str:
    return str(app_settings.storage.summaries_dir / event_cache_key / "cluster_summaries.parquet")



def sdg_summaries_output_path(
    app_settings: AppSettings,
    event_cache_key: str,
) -> str:
    return str(app_settings.storage.summaries_dir / event_cache_key / "sdg_summaries.parquet")



def executive_summary_output_path(
    app_settings: AppSettings,
    event_cache_key: str,
) -> str:
    return str(app_settings.storage.summaries_dir / event_cache_key / "executive_summary.parquet")



def sdg_executive_summary_output_path(
    app_settings: AppSettings,
    event_cache_key: str,
) -> str:
    return str(app_settings.storage.summaries_dir / event_cache_key / "sdg_executive_summary.parquet")



def persisted_cluster_summaries(
    cluster_summaries: list[SummaryRecord],
    summaries_output_path: str,
) -> str:
    return str(write_records(Path(summaries_output_path), cluster_summaries))



def persisted_sdg_summaries(
    sdg_summaries: list[SummaryRecord],
    sdg_summaries_output_path: str,
) -> str:
    return str(write_records(Path(sdg_summaries_output_path), sdg_summaries))



def persisted_executive_summary(
    executive_summary: SummaryRecord,
    executive_summary_output_path: str,
) -> str:
    return str(write_records(Path(executive_summary_output_path), [executive_summary]))



def persisted_sdg_executive_summary(
    sdg_executive_summary: SummaryRecord,
    sdg_executive_summary_output_path: str,
) -> str:
    return str(write_records(Path(sdg_executive_summary_output_path), [sdg_executive_summary]))



def persisted_summary_artifacts(
    persisted_cluster_summaries: str,
    persisted_sdg_summaries: str,
    persisted_executive_summary: str,
    persisted_sdg_executive_summary: str,
) -> dict[str, str]:
    return {
        "cluster_summaries": persisted_cluster_summaries,
        "sdg_summaries": persisted_sdg_summaries,
        "executive_summary": persisted_executive_summary,
        "sdg_executive_summary": persisted_sdg_executive_summary,
    }
