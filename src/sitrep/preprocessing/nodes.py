from __future__ import annotations

from pathlib import Path

from sitrep.preprocessing.paragraph_splitter import split_document_into_paragraphs
from sitrep.settings import AppSettings
from sitrep.storage.parquet_io import write_records
from sitrep.types import DocumentRecord, ParagraphRecord


def paragraph_records(documents_manifest: list[DocumentRecord]) -> list[ParagraphRecord]:
    paragraphs: list[ParagraphRecord] = []
    for document in documents_manifest:
        paragraphs.extend(split_document_into_paragraphs(document))
    return paragraphs


def paragraph_stats(
    paragraph_records: list[ParagraphRecord],
    documents_manifest: list[DocumentRecord],
) -> dict[str, int | float]:
    if not paragraph_records:
        return {
            "paragraph_records": 0,
            "documents_with_paragraphs": 0,
            "avg_paragraphs_per_document": 0.0,
            "avg_paragraph_chars": 0.0,
        }
    paragraph_count = len(paragraph_records)
    doc_ids = {paragraph.document_id for paragraph in paragraph_records}
    avg_per_doc = paragraph_count / max(len(doc_ids), 1)
    avg_chars = sum(len(paragraph.text) for paragraph in paragraph_records) / paragraph_count
    return {
        "paragraph_records": paragraph_count,
        "documents_with_paragraphs": len(doc_ids),
        "avg_paragraphs_per_document": round(avg_per_doc, 2),
        "avg_paragraph_chars": round(avg_chars, 2),
    }


def paragraphs_output_path(
    app_settings: AppSettings,
    event_cache_key: str,
) -> str:
    return str(app_settings.storage.paragraphs_dir / event_cache_key / "paragraph_records.parquet")


def persisted_paragraph_records(
    paragraph_records: list[ParagraphRecord],
    paragraphs_output_path: str,
) -> str:
    return str(write_records(Path(paragraphs_output_path), paragraph_records))


def persisted_preprocessing_artifacts(
    persisted_paragraph_records: str,
) -> dict[str, str]:
    return {
        "paragraph_records": persisted_paragraph_records,
    }
