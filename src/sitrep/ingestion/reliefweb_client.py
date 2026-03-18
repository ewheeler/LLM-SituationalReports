from __future__ import annotations

import json
import re
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from sitrep.ingestion.attachment_parser import extract_text_from_path, normalize_whitespace, strip_html
from sitrep.ingestion.base import AttachmentConnectorMixin, BaseConnector, SourceAccessError
from sitrep.ingestion.query_builder import build_reliefweb_request_params
from sitrep.types import DownloadedDocument, EventQuery, ExtractedContent, RemoteDocument, SourceRequest


class ReliefWebAccessError(SourceAccessError):
    pass


class ReliefWebConnector(AttachmentConnectorMixin, BaseConnector):
    def build_requests(self, query: EventQuery) -> list[SourceRequest]:
        return [
            SourceRequest(
                source_name=self.source_name,
                query=query,
                params={
                    "endpoint": self._base_url,
                    "json_body": build_reliefweb_request_params(
                        query,
                        page_size=self.settings.page_size,
                    ),
                },
            )
        ]

    def search(self, request: SourceRequest) -> list[RemoteDocument]:
        appname = self._require_appname()
        event_key = self._event_cache_key(request.query)
        search_dir = Path("search") / event_key
        remote_documents: list[RemoteDocument] = []

        for page_number in range(self.settings.max_pages):
            offset = page_number * self.settings.page_size
            request_body = deepcopy(request.params["json_body"])
            request_body["offset"] = offset
            request_path = search_dir / f"request-page-{page_number + 1:04d}.json"
            response_path = search_dir / f"response-page-{page_number + 1:04d}.json"
            self.write_json_payload(request_path, request_body)

            response_payload = self._request_or_cached_json(
                cache_path=response_path,
                method="POST",
                url=self._base_url,
                appname=appname,
                json_body=request_body,
            )

            items = response_payload.get("data", [])
            if not isinstance(items, list) or not items:
                break

            remote_documents.extend(
                self._to_remote_document(
                    item=item,
                    event_query=request.query,
                    response_path=response_path,
                )
                for item in items
            )

            total_count = int(response_payload.get("count") or 0)
            if len(items) < self.settings.page_size:
                break
            if total_count and offset + len(items) >= total_count:
                break

        return remote_documents

    def fetch(self, document: RemoteDocument) -> DownloadedDocument:
        appname = self._require_appname()
        item_path = Path("items") / f"{document.external_id}.json"
        item_response = self._request_or_cached_json(
            cache_path=item_path,
            method="GET",
            url=f"{self._base_url.rstrip('/')}/{document.external_id}",
            appname=appname,
        )
        stored_item_path = self.raw_source_dir / item_path
        fields = self._extract_fields(item_response)

        body_text = self._coalesce_text(fields.get("body-html"), fields.get("body"))
        attachment_url, attachment_name, attachment_mimetype = self._primary_attachment(fields)
        local_path = None
        body_bytes = None

        if not body_text and attachment_url:
            download_name = attachment_name or f"{document.external_id}.bin"
            download_path = Path("downloads") / document.external_id / download_name
            body_bytes, local_path = self._download_or_cached_binary(download_path, attachment_url)

        return DownloadedDocument(
            source_name=self.source_name,
            remote_document=document,
            retrieved_url=self._coalesce_text(fields.get("url_alias"), fields.get("url"), document.canonical_url),
            content_type=attachment_mimetype or ("text/html" if body_text else "application/octet-stream"),
            body_text=body_text,
            body_bytes=body_bytes,
            local_path=local_path,
            metadata={
                **document.metadata,
                "reliefweb_item_raw_path": str(stored_item_path),
                "reliefweb_attachment_url": attachment_url,
                "reliefweb_attachment_name": attachment_name,
            },
        )

    def extract_text(self, document: DownloadedDocument) -> ExtractedContent:
        attachments = ()
        attachment_url = document.metadata.get("reliefweb_attachment_url")
        if attachment_url:
            attachments = (str(attachment_url),)

        if document.body_text:
            return ExtractedContent(
                text=normalize_whitespace(strip_html(document.body_text)),
                extraction_method="reliefweb_body_html",
                attachments=attachments,
                metadata={
                    "reliefweb_local_path": str(document.local_path) if document.local_path else None,
                },
            )

        if document.local_path is not None and document.local_path.exists():
            extracted_text, method = extract_text_from_path(document.local_path)
            return ExtractedContent(
                text=normalize_whitespace(extracted_text),
                extraction_method=method,
                attachments=attachments,
                metadata={
                    "reliefweb_local_path": str(document.local_path),
                },
            )

        return ExtractedContent(
            text="",
            extraction_method="empty",
            attachments=attachments,
            metadata={
                "reliefweb_local_path": str(document.local_path) if document.local_path else None,
            },
        )

    @property
    def _base_url(self) -> str:
        return self.settings.base_url or "https://api.reliefweb.int/v2/reports"

    def _require_appname(self) -> str:
        appname = self.settings.appname
        if appname:
            return appname
        raise ReliefWebAccessError(
            "ReliefWeb requires an approved appname. Set RELIEFWEB_APPNAME or configure sources.reliefweb.appname."
        )

    def _request_or_cached_json(
        self,
        cache_path: Path,
        method: str,
        url: str,
        appname: str,
        json_body: dict[str, object] | None = None,
    ) -> dict[str, Any]:
        absolute_path = self.raw_source_dir / cache_path
        if absolute_path.exists() and (self.cache_enabled or self.offline_mode):
            payload = self.read_json_payload(cache_path)
            if isinstance(payload, dict):
                return payload
        if self.offline_mode:
            raise ReliefWebAccessError(f"Offline mode enabled and cache miss for {cache_path}")
        payload = self._request_json(method=method, url=url, appname=appname, json_body=json_body)
        self.write_json_payload(cache_path, payload)
        return payload

    def _download_or_cached_binary(self, cache_path: Path, url: str) -> tuple[bytes, Path]:
        absolute_path = self.raw_source_dir / cache_path
        if absolute_path.exists() and (self.cache_enabled or self.offline_mode):
            return absolute_path.read_bytes(), absolute_path
        if self.offline_mode:
            raise ReliefWebAccessError(f"Offline mode enabled and cache miss for {cache_path}")
        payload = self._download_binary(url)
        stored_path = self.write_binary_payload(cache_path, payload)
        return payload, stored_path

    def _request_json(
        self,
        method: str,
        url: str,
        appname: str,
        json_body: dict[str, object] | None = None,
    ) -> dict[str, Any]:
        request_url = self._with_appname(url, appname)
        request_kwargs: dict[str, Any] = {
            "method": method,
            "headers": {
                "Accept": "application/json",
                "User-Agent": self.settings.user_agent,
            },
        }
        if json_body is not None:
            request_kwargs["data"] = json.dumps(json_body).encode("utf-8")
            request_kwargs["headers"]["Content-Type"] = "application/json"

        request = Request(request_url, **request_kwargs)
        try:
            with urlopen(request, timeout=self.settings.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            response_text = error.read().decode("utf-8", errors="replace")
            message = self._error_message(response_text)
            raise ReliefWebAccessError(
                f"ReliefWeb {method} request failed with HTTP {error.code}: {message}"
            ) from error
        except URLError as error:
            raise ReliefWebAccessError(f"ReliefWeb request failed: {error.reason}") from error

    def _download_binary(self, url: str) -> bytes:
        request = Request(
            url,
            headers={
                "User-Agent": self.settings.user_agent,
            },
        )
        try:
            with urlopen(request, timeout=self.settings.timeout_seconds) as response:
                return response.read()
        except HTTPError as error:
            raise ReliefWebAccessError(
                f"ReliefWeb attachment download failed with HTTP {error.code}: {url}"
            ) from error
        except URLError as error:
            raise ReliefWebAccessError(f"ReliefWeb attachment download failed: {error.reason}") from error

    def _to_remote_document(
        self,
        item: dict[str, Any],
        event_query: EventQuery,
        response_path: Path,
    ) -> RemoteDocument:
        fields = item.get("fields", {}) if isinstance(item, dict) else {}
        external_id = str(item.get("id") or fields.get("id") or "")
        title = self._coalesce_text(fields.get("title"), f"ReliefWeb report {external_id}")
        canonical_url = self._coalesce_text(fields.get("url_alias"), fields.get("url"), f"https://reliefweb.int/node/{external_id}")
        source_names = self._names_from_objects(fields.get("source"), keys=("shortname", "name"))
        country_codes = self._names_from_objects(fields.get("country"), keys=("iso3",))
        country_names = self._names_from_objects(fields.get("country"), keys=("name",))
        tags = tuple(
            [
                *self._names_from_objects(fields.get("theme"), keys=("name",)),
                *self._names_from_objects(fields.get("disaster"), keys=("name",)),
            ]
        )
        attachments = tuple(self._file_values(fields.get("file"), "url"))
        return RemoteDocument(
            source_name=self.source_name,
            external_id=external_id,
            title=title,
            canonical_url=canonical_url,
            published_at=self._parse_datetime(self._deep_get(fields, "date", "created")),
            language=self._coalesce_text(self._deep_get_first(fields.get("language"), "name"), "en") or "en",
            metadata={
                "publisher": ", ".join(source_names) if source_names else self.source_name,
                "country_codes": tuple(country_codes),
                "country_names": tuple(country_names),
                "event_label": event_query.event_name,
                "tags": tags,
                "attachments": attachments,
                "reliefweb_raw_search_path": str(self.raw_source_dir / response_path),
            },
        )

    def _extract_fields(self, payload: dict[str, Any]) -> dict[str, Any]:
        data = payload.get("data")
        if isinstance(data, list) and data:
            return data[0].get("fields", {}) or {}
        if isinstance(data, dict):
            return data.get("fields", {}) or {}
        return {}

    def _primary_attachment(self, fields: dict[str, Any]) -> tuple[str | None, str | None, str | None]:
        files = fields.get("file")
        if not isinstance(files, list) or not files:
            return None, None, None
        first = files[0] or {}
        return (
            self._coalesce_text(first.get("url")),
            self._coalesce_text(first.get("filename")),
            self._coalesce_text(first.get("mimetype")),
        )

    def _file_values(self, files: Any, key: str) -> list[str]:
        if not isinstance(files, list):
            return []
        values: list[str] = []
        for file_entry in files:
            if not isinstance(file_entry, dict):
                continue
            value = self._coalesce_text(file_entry.get(key))
            if value:
                values.append(value)
        return values

    def _names_from_objects(self, value: Any, keys: tuple[str, ...]) -> tuple[str, ...]:
        if not isinstance(value, list):
            return ()
        results: list[str] = []
        for item in value:
            if not isinstance(item, dict):
                continue
            for key in keys:
                text = self._coalesce_text(item.get(key))
                if text:
                    results.append(text)
                    break
        return tuple(results)

    def _deep_get(self, payload: Any, *keys: str) -> Any:
        current = payload
        for key in keys:
            if not isinstance(current, dict):
                return None
            current = current.get(key)
        return current

    def _deep_get_first(self, payload: Any, key: str) -> Any:
        if isinstance(payload, list) and payload:
            first = payload[0]
            if isinstance(first, dict):
                return first.get(key)
        return None

    def _parse_datetime(self, value: Any) -> datetime | None:
        if not value or not isinstance(value, str):
            return None
        normalized = value.replace("Z", "+00:00")
        try:
            return datetime.fromisoformat(normalized)
        except ValueError:
            return None

    def _coalesce_text(self, *values: Any) -> str | None:
        for value in values:
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None

    def _event_cache_key(self, event_query: EventQuery) -> str:
        slug = re.sub(r"[^a-z0-9]+", "-", event_query.event_name.lower()).strip("-") or "event"
        countries = "-".join(country.lower() for country in event_query.country_codes) or "all"
        return f"{slug}_{countries}_{event_query.date_from.isoformat()}_{event_query.date_to.isoformat()}"

    def _with_appname(self, url: str, appname: str) -> str:
        separator = "&" if "?" in url else "?"
        return f"{url}{separator}{urlencode({'appname': appname})}"

    def _error_message(self, response_text: str) -> str:
        try:
            payload = json.loads(response_text)
        except json.JSONDecodeError:
            return response_text.strip() or "unknown ReliefWeb error"
        error = payload.get("error", {}) if isinstance(payload, dict) else {}
        if isinstance(error, dict):
            message = error.get("message")
            if isinstance(message, str) and message.strip():
                return message.strip()
        return response_text.strip() or "unknown ReliefWeb error"
