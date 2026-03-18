from __future__ import annotations

import hashlib
import re
from datetime import datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from sitrep.ingestion.attachment_parser import normalize_whitespace, strip_html
from sitrep.ingestion.base import BaseConnector, SourceAccessError
from sitrep.types import DownloadedDocument, DocumentRecord, EventQuery, ExtractedContent, RemoteDocument, SourceRequest

TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)
H1_RE = re.compile(r"<h1[^>]*>(.*?)</h1>", re.IGNORECASE | re.DOTALL)


class GenericHtmlConnector(BaseConnector):
    def build_requests(self, query: EventQuery) -> list[SourceRequest]:
        return [
            SourceRequest(
                source_name=self.source_name,
                query=query,
                params={
                    "include_urls": list(query.include_urls),
                    "exclude_urls": list(query.exclude_urls),
                },
            )
        ]

    def search(self, request: SourceRequest) -> list[RemoteDocument]:
        include_urls = [url for url in request.params.get("include_urls", []) if isinstance(url, str)]
        exclude_urls = {url for url in request.params.get("exclude_urls", []) if isinstance(url, str)}
        remote_documents: list[RemoteDocument] = []
        for index, url in enumerate(include_urls):
            if url in exclude_urls:
                continue
            remote_documents.append(
                RemoteDocument(
                    source_name=self.source_name,
                    external_id=hashlib.sha1(url.encode("utf-8")).hexdigest()[:16],
                    title=f"{self.source_name} url {index + 1}",
                    canonical_url=url,
                    published_at=None,
                    metadata={
                        "publisher": self.source_name,
                        "country_codes": request.query.country_codes,
                        "event_label": request.query.event_name,
                        "manually_curated": True,
                    },
                )
            )
        return remote_documents

    def fetch(self, document: RemoteDocument) -> DownloadedDocument:
        cache_name = hashlib.sha1(document.canonical_url.encode("utf-8")).hexdigest()
        cache_path = Path("items") / f"{cache_name}.html"
        html_text = self._fetch_or_cached_text(cache_path, document.canonical_url)
        local_path = self.raw_source_dir / cache_path
        extracted_title = self._extract_title(html_text)
        return DownloadedDocument(
            source_name=self.source_name,
            remote_document=document,
            retrieved_url=document.canonical_url,
            content_type="text/html",
            body_text=html_text,
            local_path=local_path,
            metadata={
                **document.metadata,
                "generic_html_raw_path": str(local_path),
                "html_title": extracted_title,
            },
        )

    def extract_text(self, document: DownloadedDocument) -> ExtractedContent:
        text = normalize_whitespace(strip_html(document.body_text or ""))
        return ExtractedContent(
            text=text,
            extraction_method="generic_html",
            metadata={
                "generic_html_raw_path": str(document.local_path) if document.local_path else None,
                "html_title": document.metadata.get("html_title"),
            },
        )

    def normalize(
        self,
        document: DownloadedDocument,
        content: ExtractedContent,
    ) -> DocumentRecord:
        record = super().normalize(document, content)
        html_title = content.metadata.get("html_title") or document.metadata.get("html_title")
        if isinstance(html_title, str) and html_title.strip() and record.title.startswith(f"{self.source_name} url"):
            record.title = html_title.strip()
            record.title_hash = hashlib.sha1(record.title.lower().encode("utf-8")).hexdigest()
        return record

    def _fetch_or_cached_text(self, cache_path: Path, url: str) -> str:
        absolute_path = self.raw_source_dir / cache_path
        if absolute_path.exists() and (self.cache_enabled or self.offline_mode):
            return self.read_text_payload(cache_path)
        if self.offline_mode:
            raise SourceAccessError(f"Offline mode enabled and cache miss for {cache_path}")
        text = self._request_text(url)
        self.write_text_payload(cache_path, text)
        return text

    def _request_text(self, url: str) -> str:
        request = Request(url, headers={"User-Agent": self.settings.user_agent})
        try:
            with urlopen(request, timeout=self.settings.timeout_seconds) as response:
                charset = response.headers.get_content_charset() or "utf-8"
                return response.read().decode(charset, errors="replace")
        except HTTPError as error:
            raise SourceAccessError(
                f"{self.source_name} request failed with HTTP {error.code}: {url}"
            ) from error
        except URLError as error:
            raise SourceAccessError(f"{self.source_name} request failed: {error.reason}") from error

    def _extract_title(self, html_text: str) -> str | None:
        for pattern in (TITLE_RE, H1_RE):
            match = pattern.search(html_text or "")
            if not match:
                continue
            title = normalize_whitespace(strip_html(match.group(1)))
            if title:
                return title
        return None
