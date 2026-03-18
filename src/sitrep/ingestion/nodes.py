from __future__ import annotations

from collections import Counter
from datetime import date
from pathlib import Path

from sitrep.ingestion.base import SourceConnector
from sitrep.ingestion.deduplication import build_duplicate_edges, select_canonical_documents
from sitrep.ingestion.registry import SourceRegistry, build_default_registry
from sitrep.settings import AppSettings
from sitrep.storage.json_io import write_json
from sitrep.storage.parquet_io import write_records
from sitrep.types import (
    DocumentRecord,
    DownloadedDocument,
    DuplicateEdge,
    EventQuery,
    ExtractedDocument,
    RemoteDocument,
    SourceRequest,
)


def event_query(
    event_name: str,
    country_codes: tuple,
    date_from: date,
    date_to: date,
    keywords: tuple = (),
    themes: tuple = (),
    source_profile: str | None = None,
    sources: tuple = (),
    language_filters: tuple = ("en",),
    include_urls: tuple = (),
    exclude_urls: tuple = (),
) -> EventQuery:
    return EventQuery(
        event_name=event_name,
        country_codes=country_codes,
        date_from=date_from,
        date_to=date_to,
        keywords=keywords,
        themes=themes,
        source_profile=source_profile,
        sources=sources,
        language_filters=language_filters,
        include_urls=include_urls,
        exclude_urls=exclude_urls,
    )


def event_cache_key(event_query: EventQuery) -> str:
    slug = "-".join(filter(None, "".join(char.lower() if char.isalnum() else " " for char in event_query.event_name).split()))
    slug = slug.replace(" ", "-") or "event"
    countries = "-".join(country.lower() for country in event_query.country_codes) or "all"
    return f"{slug}_{countries}_{event_query.date_from.isoformat()}_{event_query.date_to.isoformat()}"


def source_registry() -> SourceRegistry:
    return build_default_registry()


def selected_source_names(
    app_settings: AppSettings,
    event_query: EventQuery,
) -> tuple:
    return app_settings.selected_sources(
        explicit_sources=event_query.sources,
        source_profile=event_query.source_profile,
    )


def enabled_connectors(
    source_registry: SourceRegistry,
    app_settings: AppSettings,
    selected_source_names: tuple[str, ...],
) -> list[SourceConnector]:
    return source_registry.build_enabled_connectors(app_settings, selected_source_names)


def source_requests(
    enabled_connectors: list[SourceConnector],
    event_query: EventQuery,
) -> list[SourceRequest]:
    requests: list[SourceRequest] = []
    for connector in enabled_connectors:
        requests.extend(connector.build_requests(event_query))
    return requests


def remote_documents(
    enabled_connectors: list[SourceConnector],
    source_requests: list[SourceRequest],
) -> list[RemoteDocument]:
    connectors_by_name = {
        connector.source_name: connector
        for connector in enabled_connectors
    }
    documents: list[RemoteDocument] = []
    for request in source_requests:
        connector = connectors_by_name[request.source_name]
        documents.extend(connector.search(request))
    return documents


def downloaded_documents(
    enabled_connectors: list[SourceConnector],
    remote_documents: list[RemoteDocument],
) -> list[DownloadedDocument]:
    connectors_by_name = {
        connector.source_name: connector
        for connector in enabled_connectors
    }
    return [
        connectors_by_name[document.source_name].fetch(document)
        for document in remote_documents
    ]


def extracted_documents(
    enabled_connectors: list[SourceConnector],
    downloaded_documents: list[DownloadedDocument],
) -> list[ExtractedDocument]:
    connectors_by_name = {
        connector.source_name: connector
        for connector in enabled_connectors
    }
    extracted: list[ExtractedDocument] = []
    for downloaded_document in downloaded_documents:
        connector = connectors_by_name[downloaded_document.source_name]
        content = connector.extract_text(downloaded_document)
        extracted.append(
            ExtractedDocument(
                downloaded_document=downloaded_document,
                content=content,
            )
        )
    return extracted


def normalized_documents(
    enabled_connectors: list[SourceConnector],
    extracted_documents: list[ExtractedDocument],
) -> list[DocumentRecord]:
    connectors_by_name = {
        connector.source_name: connector
        for connector in enabled_connectors
    }
    normalized: list[DocumentRecord] = []
    for extracted_document in extracted_documents:
        connector = connectors_by_name[extracted_document.downloaded_document.source_name]
        normalized.append(
            connector.normalize(
                extracted_document.downloaded_document,
                extracted_document.content,
            )
        )
    return normalized


def duplicate_edges(
    normalized_documents: list[DocumentRecord],
) -> list[DuplicateEdge]:
    return build_duplicate_edges(normalized_documents)


def documents_manifest(
    normalized_documents: list[DocumentRecord],
    duplicate_edges: list[DuplicateEdge],
) -> list[DocumentRecord]:
    return select_canonical_documents(normalized_documents, duplicate_edges)


def source_mix(documents_manifest: list[DocumentRecord]) -> dict[str, int]:
    counts = Counter(document.source_connector for document in documents_manifest)
    return dict(sorted(counts.items()))


def ingestion_stats(
    remote_documents: list[RemoteDocument],
    downloaded_documents: list[DownloadedDocument],
    extracted_documents: list[ExtractedDocument],
    normalized_documents: list[DocumentRecord],
    documents_manifest: list[DocumentRecord],
    duplicate_edges: list[DuplicateEdge],
) -> dict[str, int]:
    return {
        "remote_documents": len(remote_documents),
        "downloaded_documents": len(downloaded_documents),
        "extracted_documents": len(extracted_documents),
        "normalized_documents": len(normalized_documents),
        "documents_manifest": len(documents_manifest),
        "duplicate_edges": len(duplicate_edges),
        "documents_with_text": sum(1 for document in normalized_documents if document.clean_text),
    }


def normalized_documents_output_path(
    app_settings: AppSettings,
    event_cache_key: str,
) -> str:
    return str(app_settings.storage.normalized_documents_dir / event_cache_key / "normalized_documents.parquet")


def duplicate_edges_output_path(
    app_settings: AppSettings,
    event_cache_key: str,
) -> str:
    duplicates_path = app_settings.storage.duplicates_path
    return str(duplicates_path.with_name(f"{duplicates_path.stem}-{event_cache_key}{duplicates_path.suffix}"))


def documents_manifest_output_path(
    app_settings: AppSettings,
    event_cache_key: str,
) -> str:
    manifest_path = app_settings.storage.documents_manifest_path
    return str(manifest_path.with_name(f"{manifest_path.stem}-{event_cache_key}{manifest_path.suffix}"))


def source_mix_output_path(
    app_settings: AppSettings,
    event_cache_key: str,
) -> str:
    return str(app_settings.storage.artifacts_dir / "ingestion" / f"source-mix-{event_cache_key}.json")


def run_manifest_output_path(
    app_settings: AppSettings,
    event_cache_key: str,
) -> str:
    return str(app_settings.storage.artifacts_dir / "ingestion" / f"run-manifest-{event_cache_key}.json")


def persisted_normalized_documents(
    normalized_documents: list[DocumentRecord],
    normalized_documents_output_path: str,
) -> str:
    return str(write_records(Path(normalized_documents_output_path), normalized_documents))


def persisted_duplicate_edges(
    duplicate_edges: list[DuplicateEdge],
    duplicate_edges_output_path: str,
) -> str:
    return str(write_records(Path(duplicate_edges_output_path), duplicate_edges))


def persisted_documents_manifest(
    documents_manifest: list[DocumentRecord],
    documents_manifest_output_path: str,
) -> str:
    return str(write_records(Path(documents_manifest_output_path), documents_manifest))


def persisted_source_mix(
    source_mix: dict[str, int],
    source_mix_output_path: str,
) -> str:
    return str(write_json(Path(source_mix_output_path), source_mix))


def run_manifest(
    app_settings: AppSettings,
    event_query: EventQuery,
    selected_source_names: tuple,
    ingestion_stats: dict[str, int],
    paragraph_stats: dict[str, int | float],
    cluster_stats: dict[str, int | float | str],
    question_stats: dict[str, int | float],
    sdg_stats: dict[str, int | float | dict[str, int]],
    answer_stats: dict[str, int | float],
    summary_stats: dict[str, int | float],
    persisted_ingestion_artifacts: dict[str, str],
    persisted_preprocessing_artifacts: dict[str, str],
    persisted_clustering_artifacts: dict[str, str],
    persisted_question_artifacts: dict[str, str],
    persisted_sdg_artifacts: dict[str, str],
    persisted_answer_artifacts: dict[str, str],
    persisted_summary_artifacts: dict[str, str],
    persisted_report_artifacts: dict[str, str],
) -> dict[str, object]:
    return {
        "event_name": event_query.event_name,
        "country_codes": list(event_query.country_codes),
        "date_from": event_query.date_from.isoformat(),
        "date_to": event_query.date_to.isoformat(),
        "selected_sources": list(selected_source_names),
        "include_urls": list(event_query.include_urls),
        "exclude_urls": list(event_query.exclude_urls),
        "retrieval": app_settings.to_dict().get("retrieval", {}),
        "clustering": app_settings.to_dict().get("clustering", {}),
        "generation": app_settings.to_dict().get("generation", {}),
        "sdg": app_settings.to_dict().get("sdg", {}),
        "ingestion_stats": ingestion_stats,
        "paragraph_stats": paragraph_stats,
        "cluster_stats": cluster_stats,
        "question_stats": question_stats,
        "sdg_stats": sdg_stats,
        "answer_stats": answer_stats,
        "summary_stats": summary_stats,
        "artifacts": {
            **persisted_ingestion_artifacts,
            **persisted_preprocessing_artifacts,
            **persisted_clustering_artifacts,
            **persisted_question_artifacts,
            **persisted_sdg_artifacts,
            **persisted_answer_artifacts,
            **persisted_summary_artifacts,
            **persisted_report_artifacts,
        },
    }



def persisted_run_manifest(
    run_manifest: dict[str, object],
    run_manifest_output_path: str,
) -> str:
    return str(write_json(Path(run_manifest_output_path), run_manifest))


def persisted_ingestion_artifacts(
    persisted_normalized_documents: str,
    persisted_duplicate_edges: str,
    persisted_documents_manifest: str,
    persisted_source_mix: str,
) -> dict[str, str]:
    return {
        "normalized_documents": persisted_normalized_documents,
        "duplicate_edges": persisted_duplicate_edges,
        "documents_manifest": persisted_documents_manifest,
        "source_mix": persisted_source_mix,
    }
