from __future__ import annotations

import argparse
import json
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

from sitrep.ingestion.base import SourceAccessError
from sitrep.ingestion.reliefweb_client import ReliefWebAccessError
from sitrep.llm.registry import list_prompt_specs
from sitrep.llm.runtime import GenerationProviderError
from sitrep.logging import configure_logging
from sitrep.orchestration.hamilton_driver import HamiltonUnavailableError, execute_profile
from sitrep.retrieval.hybrid import retrieve_hybrid_top_k
from sitrep.sdg.runtime import SdgProviderError
from sitrep.settings import AppSettings, load_settings
from sitrep.storage.parquet_io import load_records
from sitrep.types import ParagraphRecord


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sitrep")
    subparsers = parser.add_subparsers(dest="command", required=True)

    show_config_parser = subparsers.add_parser("show-config", help="Print resolved settings")
    show_config_parser.add_argument("--config", help="Path to a JSON, TOML, or YAML settings file")
    _add_runtime_override_args(show_config_parser)

    prompts_parser = subparsers.add_parser("list-prompts", help="List registered prompt templates")
    prompts_parser.add_argument("--json", action="store_true", help="Print prompt metadata as JSON")

    ingest_parser = subparsers.add_parser("ingest", help="Run the ingestion scaffold")
    _add_run_args(ingest_parser, profile_default="ingest_only")

    run_parser = subparsers.add_parser("run", help="Run a configured profile")
    _add_run_args(run_parser, profile_default="ingest_only")

    retrieve_parser = subparsers.add_parser("retrieve", help="Run lexical retrieval against paragraph artifacts")
    retrieve_parser.add_argument("--paragraphs-path", required=True)
    retrieve_parser.add_argument("--query", required=True)
    retrieve_parser.add_argument("--top-k", type=int, default=5)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging()

    if args.command == "list-prompts":
        prompts = list_prompt_specs()
        if args.json:
            print(json.dumps(prompts, indent=2))
        else:
            for prompt in prompts:
                print(f"{prompt['prompt_id']}	{prompt['schema_name']}	{prompt['default_model']}	{prompt['description']}")
        return 0

    if args.command == "retrieve":
        records = load_records(Path(args.paragraphs_path))
        paragraph_records = [_paragraph_record_from_dict(record) for record in records]
        settings = load_settings(None)
        hits = retrieve_hybrid_top_k(
            args.query,
            paragraph_records,
            top_k=args.top_k,
            lexical_weight=settings.retrieval.lexical_weight,
            title_weight=settings.retrieval.title_weight,
            trust_weight=settings.retrieval.trust_weight,
            recency_weight=settings.retrieval.recency_weight,
        )
        print(json.dumps(_json_safe(hits), indent=2, default=str))
        return 0

    settings = load_settings(getattr(args, "config", None))
    settings = settings.with_runtime_overrides(
        offline_mode=True if getattr(args, "offline_mode", False) else None,
        cache_enabled=False if getattr(args, "disable_cache", False) else None,
        reliefweb_appname=getattr(args, "reliefweb_appname", None),
        fixture_manifest_path=getattr(args, "fixture_manifest", None),
    )

    if args.command == "show-config":
        print(json.dumps(settings.to_dict(), indent=2))
        return 0

    if args.command in {"run", "ingest"}:
        payload = _run_payload(args, settings)
        if args.dry_run:
            print(json.dumps(payload, indent=2, default=str))
            return 0

        try:
            results = execute_profile(
                app_settings=settings,
                event_name=payload["event_name"],
                country_codes=tuple(payload["country_codes"]),
                date_from=payload["date_from"],
                date_to=payload["date_to"],
                keywords=tuple(payload["keywords"]),
                themes=tuple(payload["themes"]),
                source_profile=payload["source_profile"],
                sources=tuple(payload["sources"]),
                include_urls=tuple(payload["include_urls"]),
                exclude_urls=tuple(payload["exclude_urls"]),
                profile_name=payload["profile_name"],
            )
        except (
            GenerationProviderError,
            HamiltonUnavailableError,
            ReliefWebAccessError,
            SdgProviderError,
            SourceAccessError,
        ) as error:
            parser.error(str(error))
            return 2

        print(json.dumps(_json_safe(results), indent=2, default=str))
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


def _add_run_args(parser: argparse.ArgumentParser, *, profile_default: str) -> None:
    parser.add_argument("--config", help="Path to a JSON, TOML, or YAML settings file")
    parser.add_argument("--profile", default=profile_default)
    parser.add_argument("--event", required=True)
    parser.add_argument("--country", action="append", dest="countries", required=True)
    parser.add_argument("--date-from", required=True)
    parser.add_argument("--date-to", required=True)
    parser.add_argument("--keyword", action="append", dest="keywords", default=[])
    parser.add_argument("--theme", action="append", dest="themes", default=[])
    parser.add_argument("--source-profile")
    parser.add_argument("--sources", help="Comma-separated source names")
    parser.add_argument("--include-url", action="append", dest="include_urls", default=[])
    parser.add_argument("--exclude-url", action="append", dest="exclude_urls", default=[])
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the resolved execution inputs without invoking Hamilton.",
    )
    _add_runtime_override_args(parser)


def _add_runtime_override_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--offline-mode",
        action="store_true",
        help="Require cached source responses instead of making network calls.",
    )
    parser.add_argument(
        "--disable-cache",
        action="store_true",
        help="Force fresh source fetches instead of reusing cached responses.",
    )
    parser.add_argument(
        "--reliefweb-appname",
        help="Approved ReliefWeb appname to use for API requests.",
    )
    parser.add_argument(
        "--fixture-manifest",
        help="Path to a local fixture manifest JSON file for the fixtures connector.",
    )


def _run_payload(args: argparse.Namespace, settings: AppSettings) -> dict[str, Any]:
    explicit_sources = tuple(
        source.strip()
        for source in (args.sources or "").split(",")
        if source.strip()
    )
    return {
        "profile_name": args.profile,
        "event_name": args.event,
        "country_codes": list(args.countries),
        "date_from": date.fromisoformat(args.date_from),
        "date_to": date.fromisoformat(args.date_to),
        "keywords": list(args.keywords),
        "themes": list(args.themes),
        "source_profile": args.source_profile or settings.run.default_source_profile,
        "sources": list(explicit_sources),
        "include_urls": list(args.include_urls),
        "exclude_urls": list(args.exclude_urls),
        "resolved_sources": list(
            settings.selected_sources(
                explicit_sources=explicit_sources,
                source_profile=args.source_profile,
            )
        ),
        "offline_mode": settings.run.offline_mode,
        "cache_enabled": settings.run.cache_enabled,
    }


def _paragraph_record_from_dict(record: dict[str, Any]) -> ParagraphRecord:
    published_at = record.get("published_at")
    if isinstance(published_at, str) and published_at:
        try:
            published_at = datetime.fromisoformat(published_at)
        except ValueError:
            published_at = None
    elif not isinstance(published_at, datetime):
        published_at = None
    return ParagraphRecord(
        paragraph_id=record["paragraph_id"],
        document_id=record["document_id"],
        paragraph_index=int(record["paragraph_index"]),
        text=record["text"],
        start_char=int(record["start_char"]),
        end_char=int(record["end_char"]),
        source_connector=record["source_connector"],
        publisher=record["publisher"],
        canonical_url=record["canonical_url"],
        retrieved_url=record["retrieved_url"],
        title=record["title"],
        published_at=published_at,
        language=record["language"],
        country_codes=tuple(record.get("country_codes", [])),
        event_label=record["event_label"],
        tags=tuple(record.get("tags", [])),
        trust_tier=record.get("trust_tier", ""),
        attachments=tuple(record.get("attachments", [])),
        metadata=record.get("metadata", {}),
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if hasattr(value, "__dict__"):
        return {
            key: _json_safe(item)
            for key, item in vars(value).items()
        }
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    return value


if __name__ == "__main__":
    raise SystemExit(main())
