from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from sitrep.ingestion.attachment_parser import extract_text_from_path
from sitrep.ingestion.base import BaseConnector, SourceAccessError
from sitrep.types import DownloadedDocument, EventQuery, ExtractedContent, RemoteDocument, SourceRequest


class FixtureManifestConnector(BaseConnector):
    def build_requests(self, query: EventQuery) -> list[SourceRequest]:
        manifest_path = self.settings.fixture_manifest_path
        if not manifest_path:
            raise SourceAccessError(
                f"{self.source_name} requires 'fixture_manifest_path' in settings or --fixture-manifest at runtime."
            )
        return [
            SourceRequest(
                source_name=self.source_name,
                query=query,
                params={"fixture_manifest_path": manifest_path},
            )
        ]

    def search(self, request: SourceRequest) -> list[RemoteDocument]:
        manifest_path = Path(str(request.params["fixture_manifest_path"])).expanduser().resolve()
        if not manifest_path.exists():
            raise SourceAccessError(f"Fixture manifest not found: {manifest_path}")
        payload = json.loads(manifest_path.read_text())
        entries = self._manifest_entries(payload)
        manifest_snapshot = Path("manifests") / f"{manifest_path.stem}.json"
        self.write_json_payload(manifest_snapshot, payload)
        documents: list[RemoteDocument] = []
        for index, entry in enumerate(entries, start=1):
            if not isinstance(entry, dict):
                continue
            fixture_path = self._fixture_path(entry, manifest_path)
            if fixture_path is None or not fixture_path.exists():
                continue
            file_url = self._coalesce_text(entry.get("file_url")) or fixture_path.as_uri()
            original_url = self._coalesce_text(entry.get("original_url"))
            title = self._coalesce_text(entry.get("title"), fixture_path.stem.replace("-", " ").title())
            canonical_url = original_url or file_url
            if not self._selected_by_url(canonical_url, file_url, request.query):
                continue
            publisher = self._publisher_from_url(original_url or file_url)
            documents.append(
                RemoteDocument(
                    source_name=self.source_name,
                    external_id=self._external_id(entry, fixture_path, index),
                    title=title,
                    canonical_url=canonical_url,
                    published_at=self._parse_datetime(entry.get("published_at")),
                    language=self._coalesce_text(entry.get("language"), "en") or "en",
                    metadata={
                        "publisher": publisher,
                        "country_codes": request.query.country_codes,
                        "event_label": request.query.event_name,
                        "tags": tuple(request.query.themes),
                        "fixture_path": str(fixture_path),
                        "fixture_file_url": file_url,
                        "original_url": original_url,
                        "fixture_manifest_path": str(manifest_path),
                        "fixture_snapshot_path": str(self.raw_source_dir / manifest_snapshot),
                        "paragraph_count": entry.get("paragraph_count"),
                        "manually_curated": True,
                    },
                )
            )
        return documents

    def fetch(self, document: RemoteDocument) -> DownloadedDocument:
        fixture_path_value = document.metadata.get("fixture_path")
        if not isinstance(fixture_path_value, str) or not fixture_path_value:
            raise SourceAccessError(f"Fixture document is missing a local path: {document.external_id}")
        fixture_path = Path(fixture_path_value).expanduser().resolve()
        if not fixture_path.exists():
            raise SourceAccessError(f"Fixture document not found: {fixture_path}")

        content_type = self._content_type_for_suffix(fixture_path.suffix.lower())
        body_text: str | None = None
        body_bytes: bytes | None = None
        if content_type.startswith("text/") or fixture_path.suffix.lower() in {".html", ".htm", ".txt", ".md"}:
            body_text = fixture_path.read_text(errors="replace")
        else:
            body_bytes = fixture_path.read_bytes()

        return DownloadedDocument(
            source_name=self.source_name,
            remote_document=document,
            retrieved_url=str(document.metadata.get("fixture_file_url") or fixture_path.as_uri()),
            content_type=content_type,
            body_text=body_text,
            body_bytes=body_bytes,
            local_path=fixture_path,
            metadata={
                **document.metadata,
                "fixture_local_path": str(fixture_path),
            },
        )

    def extract_text(self, document: DownloadedDocument) -> ExtractedContent:
        if document.local_path is None:
            return super().extract_text(document)
        text, extraction_method = extract_text_from_path(document.local_path)
        attachment_hint: tuple[str, ...] = ()
        if document.local_path.suffix.lower() not in {".html", ".htm", ".txt", ".md"}:
            attachment_hint = (str(document.local_path),)
        return ExtractedContent(
            text=text,
            extraction_method=f"fixture_{extraction_method}",
            attachments=attachment_hint,
            metadata={
                "fixture_local_path": str(document.local_path),
                "fixture_manifest_path": document.metadata.get("fixture_manifest_path"),
            },
        )

    def _manifest_entries(self, payload: Any) -> list[dict[str, Any]]:
        if isinstance(payload, list):
            return [entry for entry in payload if isinstance(entry, dict)]
        if isinstance(payload, dict):
            for key in ("documents", "entries", "items"):
                value = payload.get(key)
                if isinstance(value, list):
                    return [entry for entry in value if isinstance(entry, dict)]
        raise SourceAccessError("Fixture manifest must be a list or contain a top-level 'documents'/'entries' list.")

    def _fixture_path(self, entry: dict[str, Any], manifest_path: Path) -> Path | None:
        fixture_path_value = self._coalesce_text(entry.get("fixture_path"))
        if fixture_path_value:
            path = Path(fixture_path_value).expanduser()
            return path.resolve() if path.is_absolute() else (manifest_path.parent / path).resolve()
        file_url = self._coalesce_text(entry.get("file_url"))
        if file_url and file_url.startswith("file://"):
            return Path(unquote(urlparse(file_url).path)).resolve()
        return None

    def _selected_by_url(self, canonical_url: str, file_url: str, query: EventQuery) -> bool:
        include_urls = {value for value in query.include_urls if value}
        exclude_urls = {value for value in query.exclude_urls if value}
        url_candidates = {candidate for candidate in (canonical_url, file_url) if candidate}
        if include_urls and not any(any(include in candidate for include in include_urls) for candidate in url_candidates):
            return False
        if exclude_urls and any(any(exclude in candidate for exclude in exclude_urls) for candidate in url_candidates):
            return False
        return True

    def _external_id(self, entry: dict[str, Any], fixture_path: Path, index: int) -> str:
        for key in ("id", "external_id"):
            value = entry.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        digest = hashlib.sha1(str(fixture_path).encode("utf-8")).hexdigest()[:12]
        return f"fixture-{index:03d}-{digest}"

    def _publisher_from_url(self, url: str | None) -> str:
        if not url:
            return self.source_name
        hostname = urlparse(url).netloc.lower().removeprefix("www.")
        if not hostname:
            return self.source_name
        parts = [segment for segment in hostname.split(".") if segment and segment not in {"com", "org", "net", "int", "co", "uk"}]
        if not parts:
            return hostname
        return " ".join(segment.replace("-", " ").title() for segment in parts[:2])

    def _parse_datetime(self, value: Any) -> datetime | None:
        if not isinstance(value, str) or not value.strip():
            return None
        normalized = value.strip().replace("Z", "+00:00")
        try:
            return datetime.fromisoformat(normalized)
        except ValueError:
            return None

    def _coalesce_text(self, *values: Any) -> str | None:
        for value in values:
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None

    def _content_type_for_suffix(self, suffix: str) -> str:
        return {
            ".html": "text/html",
            ".htm": "text/html",
            ".txt": "text/plain",
            ".md": "text/markdown",
            ".pdf": "application/pdf",
            ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        }.get(suffix, "application/octet-stream")
