from __future__ import annotations

import hashlib
import re
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from xml.etree import ElementTree

from sitrep.ingestion.attachment_parser import normalize_whitespace, strip_html
from sitrep.ingestion.base import BaseConnector, SourceAccessError
from sitrep.ingestion.query_builder import build_rss_request_params
from sitrep.types import DownloadedDocument, EventQuery, ExtractedContent, RemoteDocument, SourceRequest

ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}


class RSSConnector(BaseConnector):
    def build_requests(self, query: EventQuery) -> list[SourceRequest]:
        return [
            SourceRequest(
                source_name=self.source_name,
                query=query,
                params={
                    **build_rss_request_params(query),
                    "feed_url": self.settings.feed_url,
                },
            )
        ]

    def search(self, request: SourceRequest) -> list[RemoteDocument]:
        feed_url = request.params.get("feed_url")
        if not isinstance(feed_url, str) or not feed_url.strip():
            return []
        event_key = self._event_cache_key(request.query)
        cache_path = Path("feeds") / f"{event_key}.xml"
        xml_text = self._fetch_or_cached_text(cache_path, feed_url)
        return self._parse_feed(xml_text, request.query, cache_path)

    def fetch(self, document: RemoteDocument) -> DownloadedDocument:
        if not document.canonical_url:
            return super().fetch(document)
        cache_name = hashlib.sha1(document.canonical_url.encode("utf-8")).hexdigest()
        cache_path = Path("items") / f"{cache_name}.html"
        html_text = self._fetch_or_cached_text(cache_path, document.canonical_url)
        local_path = self.raw_source_dir / cache_path
        return DownloadedDocument(
            source_name=self.source_name,
            remote_document=document,
            retrieved_url=document.canonical_url,
            content_type="text/html",
            body_text=html_text,
            local_path=local_path,
            metadata={
                **document.metadata,
                "rss_item_raw_path": str(local_path),
            },
        )

    def extract_text(self, document: DownloadedDocument) -> ExtractedContent:
        text = normalize_whitespace(strip_html(document.body_text or ""))
        return ExtractedContent(
            text=text,
            extraction_method="rss_html",
            metadata={
                "rss_item_raw_path": str(document.local_path) if document.local_path else None,
            },
        )

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

    def _parse_feed(
        self,
        xml_text: str,
        event_query: EventQuery,
        cache_path: Path,
    ) -> list[RemoteDocument]:
        try:
            root = ElementTree.fromstring(xml_text)
        except ElementTree.ParseError:
            return []

        entries = root.findall("./channel/item")
        if not entries:
            entries = root.findall("./atom:entry", ATOM_NS)

        documents: list[RemoteDocument] = []
        query_terms = self._query_terms(event_query)
        for index, entry in enumerate(entries):
            title = self._entry_text(entry, "title") or self._entry_text(entry, "{http://www.w3.org/2005/Atom}title")
            summary = self._entry_text(entry, "description") or self._entry_text(entry, "summary") or self._entry_text(entry, "{http://www.w3.org/2005/Atom}summary")
            if not self._matches_query(title=title, summary=summary, query_terms=query_terms):
                continue
            link = self._entry_link(entry)
            if not link:
                continue
            published = self._parse_datetime(
                self._entry_text(entry, "pubDate")
                or self._entry_text(entry, "published")
                or self._entry_text(entry, "updated")
                or self._entry_text(entry, "{http://www.w3.org/2005/Atom}published")
                or self._entry_text(entry, "{http://www.w3.org/2005/Atom}updated")
            )
            documents.append(
                RemoteDocument(
                    source_name=self.source_name,
                    external_id=self._entry_id(entry, index),
                    title=title or f"{self.source_name} item {index}",
                    canonical_url=link,
                    published_at=published,
                    metadata={
                        "publisher": self.source_name,
                        "country_codes": event_query.country_codes,
                        "event_label": event_query.event_name,
                        "rss_feed_cache_path": str(self.raw_source_dir / cache_path),
                    },
                )
            )
        return documents

    def _query_terms(self, event_query: EventQuery) -> tuple[str, ...]:
        terms = [event_query.event_name, *event_query.keywords, *event_query.country_codes]
        return tuple(term.lower() for term in terms if term)

    def _matches_query(self, *, title: str | None, summary: str | None, query_terms: tuple[str, ...]) -> bool:
        if not query_terms:
            return True
        haystack = f"{title or ''} {summary or ''}".lower()
        return any(term in haystack for term in query_terms)

    def _entry_text(self, entry: ElementTree.Element, tag: str) -> str | None:
        node = entry.find(tag)
        if node is not None and node.text:
            return node.text.strip()
        return None

    def _entry_link(self, entry: ElementTree.Element) -> str | None:
        link_text = self._entry_text(entry, "link")
        if link_text:
            return link_text
        atom_link = entry.find("atom:link", ATOM_NS)
        if atom_link is not None:
            href = atom_link.attrib.get("href")
            if href:
                return href.strip()
        namespaced_link = entry.find("{http://www.w3.org/2005/Atom}link")
        if namespaced_link is not None:
            href = namespaced_link.attrib.get("href")
            if href:
                return href.strip()
        return None

    def _entry_id(self, entry: ElementTree.Element, fallback_index: int) -> str:
        guid = self._entry_text(entry, "guid") or self._entry_text(entry, "id") or self._entry_text(entry, "{http://www.w3.org/2005/Atom}id")
        if guid:
            return guid
        title = self._entry_text(entry, "title") or self._entry_text(entry, "{http://www.w3.org/2005/Atom}title") or str(fallback_index)
        return hashlib.sha1(title.encode("utf-8")).hexdigest()[:16]

    def _parse_datetime(self, value: str | None) -> datetime | None:
        if not value:
            return None
        for fmt in (
            "%a, %d %b %Y %H:%M:%S %z",
            "%Y-%m-%dT%H:%M:%S%z",
            "%Y-%m-%dT%H:%M:%SZ",
        ):
            try:
                return datetime.strptime(value, fmt)
            except ValueError:
                continue
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None

    def _event_cache_key(self, event_query: EventQuery) -> str:
        slug = re.sub(r"[^a-z0-9]+", "-", event_query.event_name.lower()).strip("-") or "event"
        countries = "-".join(country.lower() for country in event_query.country_codes) or "all"
        return f"{slug}_{countries}_{event_query.date_from.isoformat()}_{event_query.date_to.isoformat()}"
