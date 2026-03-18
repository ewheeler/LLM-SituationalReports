from __future__ import annotations

from pathlib import Path

from sitrep.reports.assemble import build_markdown_report, build_report_payload
from sitrep.reports.postprocess import finalize_report_payload
from sitrep.settings import AppSettings
from sitrep.storage.json_io import write_json
from sitrep.types import AnswerRecord, SummaryRecord


def report_payload(
    event_name: str,
    executive_summary: SummaryRecord,
    cluster_summaries: list[SummaryRecord],
    answer_records: list[AnswerRecord],
) -> dict[str, object]:
    return build_report_payload(
        event_name,
        executive_summary,
        cluster_summaries,
        answer_records,
        grouping="cluster",
    )


def finalized_report_payload(report_payload: dict[str, object]) -> dict[str, object]:
    return finalize_report_payload(report_payload, prefer_section_order=True)


def sdg_report_payload(
    event_name: str,
    sdg_executive_summary: SummaryRecord,
    sdg_summaries: list[SummaryRecord],
    answer_records: list[AnswerRecord],
) -> dict[str, object]:
    return build_report_payload(
        event_name,
        sdg_executive_summary,
        sdg_summaries,
        answer_records,
        grouping="sdg",
    )


def finalized_sdg_report_payload(sdg_report_payload: dict[str, object]) -> dict[str, object]:
    return finalize_report_payload(sdg_report_payload, prefer_section_order=True)


def markdown_report(finalized_report_payload: dict[str, object]) -> str:
    return build_markdown_report(finalized_report_payload)


def sdg_markdown_report(finalized_sdg_report_payload: dict[str, object]) -> str:
    return build_markdown_report(finalized_sdg_report_payload)


def report_json_output_path(
    app_settings: AppSettings,
    event_cache_key: str,
) -> str:
    return str(app_settings.storage.reports_dir / event_cache_key / "report.json")


def report_markdown_output_path(
    app_settings: AppSettings,
    event_cache_key: str,
) -> str:
    return str(app_settings.storage.reports_dir / event_cache_key / "report.md")


def sdg_report_json_output_path(
    app_settings: AppSettings,
    event_cache_key: str,
) -> str:
    return str(app_settings.storage.reports_dir / event_cache_key / "report_sdg.json")


def sdg_report_markdown_output_path(
    app_settings: AppSettings,
    event_cache_key: str,
) -> str:
    return str(app_settings.storage.reports_dir / event_cache_key / "report_sdg.md")


def persisted_report_json(
    finalized_report_payload: dict[str, object],
    report_json_output_path: str,
) -> str:
    return str(write_json(Path(report_json_output_path), finalized_report_payload))


def persisted_report_markdown(
    markdown_report: str,
    report_markdown_output_path: str,
) -> str:
    output_path = Path(report_markdown_output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown_report)
    return str(output_path)


def persisted_sdg_report_json(
    finalized_sdg_report_payload: dict[str, object],
    sdg_report_json_output_path: str,
) -> str:
    return str(write_json(Path(sdg_report_json_output_path), finalized_sdg_report_payload))


def persisted_sdg_report_markdown(
    sdg_markdown_report: str,
    sdg_report_markdown_output_path: str,
) -> str:
    output_path = Path(sdg_report_markdown_output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(sdg_markdown_report)
    return str(output_path)


def persisted_report_artifacts(
    persisted_report_json: str,
    persisted_report_markdown: str,
    persisted_sdg_report_json: str,
    persisted_sdg_report_markdown: str,
) -> dict[str, str]:
    return {
        "report_json": persisted_report_json,
        "report_markdown": persisted_report_markdown,
        "report_sdg_json": persisted_sdg_report_json,
        "report_sdg_markdown": persisted_sdg_report_markdown,
    }
