from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Protocol

from sitrep.settings import SourceSettings
from sitrep.types import (
    DocumentRecord,
    DownloadedDocument,
    EventQuery,
    ExtractedContent,
    RemoteDocument,
    SourceRequest,
)


class SourceAccessError(RuntimeError):
    pass


class SourceConnector(Protocol):
    source_name: str
    trust_tier: str

    def build_requests(self, query: EventQuery) -> list[SourceRequest]: ...
    def search(self, request: SourceRequest) -> list[RemoteDocument]: ...
    def fetch(self, document: RemoteDocument) -> DownloadedDocument: ...
    def extract_text(self, document: DownloadedDocument) -> ExtractedContent: ...
    def normalize(
        self,
        document: DownloadedDocument,
        content: ExtractedContent,
    ) -> DocumentRecord: ...


class BaseConnector(ABC):
    def __init__(
        self,
        source_name: str,
        settings: SourceSettings,
        raw_root_dir: Path | None = None,
        cache_enabled: bool = True,
        offline_mode: bool = False,
    ) -> None:
        self.source_name = source_name
        self.settings = settings
        self.trust_tier = settings.trust_tier.value
        self.raw_root_dir = (raw_root_dir or Path("data/raw")).resolve()
        self.raw_source_dir = self.raw_root_dir / self.source_name
        self.cache_enabled = cache_enabled
        self.offline_mode = offline_mode

    @abstractmethod
    def build_requests(self, query: EventQuery) -> list[SourceRequest]:
        raise NotImplementedError

    @abstractmethod
    def search(self, request: SourceRequest) -> list[RemoteDocument]:
        raise NotImplementedError

    def fetch(self, document: RemoteDocument) -> DownloadedDocument:
        return DownloadedDocument(
            source_name=self.source_name,
            remote_document=document,
            retrieved_url=document.canonical_url,
            metadata={"connector_status": "fetch_not_implemented"},
        )

    def extract_text(self, document: DownloadedDocument) -> ExtractedContent:
        text = document.body_text or ""
        return ExtractedContent(
            text=_normalize_whitespace(_strip_html(text)),
            extraction_method="passthrough",
            metadata={"connector_status": "extract_not_implemented"},
        )

    def normalize(
        self,
        document: DownloadedDocument,
        content: ExtractedContent,
    ) -> DocumentRecord:
        remote = document.remote_document
        clean_text = _normalize_whitespace(content.text)
        title_hash = hashlib.sha1(remote.title.lower().encode("utf-8")).hexdigest()
        content_hash = hashlib.sha1(clean_text.encode("utf-8")).hexdigest()
        document_id = hashlib.sha1(
            f"{self.source_name}:{remote.external_id}:{remote.canonical_url}".encode("utf-8")
        ).hexdigest()
        return DocumentRecord(
            document_id=document_id,
            external_id=remote.external_id,
            source_connector=self.source_name,
            publisher=remote.metadata.get("publisher", self.source_name),
            trust_tier=self.trust_tier,
            canonical_url=remote.canonical_url,
            retrieved_url=document.retrieved_url,
            title=remote.title,
            published_at=remote.published_at,
            language=remote.language,
            content_type=document.content_type,
            country_codes=tuple(remote.metadata.get("country_codes", ())),
            event_label=remote.metadata.get("event_label", ""),
            tags=tuple(remote.metadata.get("tags", ())),
            raw_text=content.text,
            clean_text=clean_text,
            attachments=content.attachments,
            content_hash=content_hash,
            title_hash=title_hash,
            license_or_terms_hint=remote.metadata.get("license_or_terms_hint"),
            retrieval_metadata={
                **remote.metadata,
                **document.metadata,
                **content.metadata,
            },
        )

    def ensure_raw_subdir(self, relative_path: str | Path = ".") -> Path:
        path = self.raw_source_dir / Path(relative_path)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def write_json_payload(self, relative_path: str | Path, payload: object) -> Path:
        output_path = self.raw_source_dir / Path(relative_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, default=str)
        )
        return output_path

    def read_json_payload(self, relative_path: str | Path) -> object:
        input_path = self.raw_source_dir / Path(relative_path)
        return json.loads(input_path.read_text())

    def write_text_payload(self, relative_path: str | Path, payload: str) -> Path:
        output_path = self.raw_source_dir / Path(relative_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload)
        return output_path

    def read_text_payload(self, relative_path: str | Path) -> str:
        input_path = self.raw_source_dir / Path(relative_path)
        return input_path.read_text()

    def write_binary_payload(self, relative_path: str | Path, payload: bytes) -> Path:
        output_path = self.raw_source_dir / Path(relative_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(payload)
        return output_path


class AttachmentConnectorMixin:
    def attachment_urls(self, document: RemoteDocument) -> tuple[str, ...]:
        attachments = document.metadata.get("attachments", ())
        return tuple(str(url) for url in attachments)



def _normalize_whitespace(text: str) -> str:
    return " ".join(text.split())



def _strip_html(text: str) -> str:
    return text.replace("<br>", " ").replace("<br/>", " ").replace("<br />", " ")
