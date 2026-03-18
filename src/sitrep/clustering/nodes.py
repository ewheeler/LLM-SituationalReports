from __future__ import annotations

from pathlib import Path

from sitrep.clustering.baseline import cluster_paragraphs
from sitrep.clustering.notebook_style import cluster_paragraphs_notebook_style
from sitrep.settings import AppSettings
from sitrep.storage.json_io import write_json
from sitrep.storage.parquet_io import write_records
from sitrep.types import ClusterAssignment, ClusterRecord, ParagraphRecord


def cluster_outputs(
    app_settings: AppSettings,
    paragraph_records: list[ParagraphRecord],
) -> dict[str, list | dict]:
    provider = app_settings.clustering.provider
    if provider == "baseline":
        clusters, assignments = cluster_paragraphs(paragraph_records)
        diagnostics = {
            "provider": provider,
            "status": "ok",
        }
    elif provider == "notebook_hdbscan":
        clusters, assignments, diagnostics = cluster_paragraphs_notebook_style(
            paragraph_records,
            app_settings.clustering,
        )
    else:
        raise ValueError(f"Unsupported clustering provider: {provider}")
    return {
        "clusters": clusters,
        "assignments": assignments,
        "diagnostics": diagnostics,
    }


def cluster_records(cluster_outputs: dict[str, list | dict]) -> list[ClusterRecord]:
    return list(cluster_outputs.get("clusters", []))


def cluster_assignments(cluster_outputs: dict[str, list | dict]) -> list[ClusterAssignment]:
    return list(cluster_outputs.get("assignments", []))


def clustering_diagnostics(cluster_outputs: dict[str, list | dict]) -> dict[str, object]:
    diagnostics = cluster_outputs.get("diagnostics", {})
    return diagnostics if isinstance(diagnostics, dict) else {}


def clustered_paragraph_records(
    paragraph_records: list[ParagraphRecord],
    cluster_assignments: list[ClusterAssignment],
) -> list[ParagraphRecord]:
    assignment_by_paragraph = {
        assignment.paragraph_id: assignment
        for assignment in cluster_assignments
    }
    updated: list[ParagraphRecord] = []
    for paragraph in paragraph_records:
        assignment = assignment_by_paragraph.get(paragraph.paragraph_id)
        metadata = dict(paragraph.metadata)
        if assignment is not None:
            metadata["cluster_id"] = assignment.cluster_id
            metadata["assignment_score"] = assignment.score
        updated.append(
            ParagraphRecord(
                paragraph_id=paragraph.paragraph_id,
                document_id=paragraph.document_id,
                paragraph_index=paragraph.paragraph_index,
                text=paragraph.text,
                start_char=paragraph.start_char,
                end_char=paragraph.end_char,
                source_connector=paragraph.source_connector,
                publisher=paragraph.publisher,
                canonical_url=paragraph.canonical_url,
                retrieved_url=paragraph.retrieved_url,
                title=paragraph.title,
                published_at=paragraph.published_at,
                language=paragraph.language,
                country_codes=paragraph.country_codes,
                event_label=paragraph.event_label,
                tags=paragraph.tags,
                trust_tier=paragraph.trust_tier,
                attachments=paragraph.attachments,
                metadata=metadata,
            )
        )
    return updated


def cluster_stats(
    cluster_records: list[ClusterRecord],
    cluster_assignments: list[ClusterAssignment],
    clustering_diagnostics: dict[str, object],
) -> dict[str, int | float | str]:
    provider = str(clustering_diagnostics.get("provider", "baseline"))
    if not cluster_records:
        return {
            "provider": provider,
            "clusters": 0,
            "cluster_assignments": len(cluster_assignments),
            "avg_paragraphs_per_cluster": 0.0,
            "largest_cluster": 0,
        }
    avg_size = sum(cluster.paragraph_count for cluster in cluster_records) / len(cluster_records)
    largest = max(cluster.paragraph_count for cluster in cluster_records)
    stats: dict[str, int | float | str] = {
        "provider": provider,
        "clusters": len(cluster_records),
        "cluster_assignments": len(cluster_assignments),
        "avg_paragraphs_per_cluster": round(avg_size, 2),
        "largest_cluster": largest,
    }
    if "best_run" in clustering_diagnostics:
        best_run = clustering_diagnostics.get("best_run", {})
        if isinstance(best_run, dict):
            if "dbcv" in best_run:
                stats["best_dbcv"] = float(best_run["dbcv"])
            if "score" in best_run:
                stats["best_score"] = float(best_run["score"])
    return stats


def cluster_records_output_path(
    app_settings: AppSettings,
    event_cache_key: str,
) -> str:
    return str(app_settings.storage.clusters_dir / event_cache_key / "cluster_records.parquet")


def cluster_assignments_output_path(
    app_settings: AppSettings,
    event_cache_key: str,
) -> str:
    return str(app_settings.storage.clusters_dir / event_cache_key / "cluster_assignments.parquet")


def clustering_diagnostics_output_path(
    app_settings: AppSettings,
    event_cache_key: str,
) -> str:
    return str(app_settings.storage.clusters_dir / event_cache_key / "clustering_diagnostics.json")


def persisted_cluster_records(
    cluster_records: list[ClusterRecord],
    cluster_records_output_path: str,
) -> str:
    return str(write_records(Path(cluster_records_output_path), cluster_records))


def persisted_cluster_assignments(
    cluster_assignments: list[ClusterAssignment],
    cluster_assignments_output_path: str,
) -> str:
    return str(write_records(Path(cluster_assignments_output_path), cluster_assignments))


def persisted_clustering_diagnostics(
    clustering_diagnostics: dict[str, object],
    clustering_diagnostics_output_path: str,
) -> str:
    return str(write_json(Path(clustering_diagnostics_output_path), clustering_diagnostics))


def persisted_clustering_artifacts(
    persisted_cluster_records: str,
    persisted_cluster_assignments: str,
    persisted_clustering_diagnostics: str,
) -> dict[str, str]:
    return {
        "cluster_records": persisted_cluster_records,
        "cluster_assignments": persisted_cluster_assignments,
        "clustering_diagnostics": persisted_clustering_diagnostics,
    }
