from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any


class SourceKind(str, Enum):
    RELIEFWEB_API = "reliefweb_api"
    RSS_HTML = "rss_html"
    GENERIC_HTML = "generic_html"
    FIXTURE_MANIFEST = "fixture_manifest"


class TrustTier(str, Enum):
    HUMANITARIAN_PRIMARY = "humanitarian_primary"
    MAJOR_NEWS = "major_news"
    CURATED_WEB = "curated_web"
    DISABLED = "disabled"


JsonDict = dict[str, Any]


@dataclass(slots=True)
class EventQuery:
    event_name: str
    country_codes: tuple[str, ...]
    date_from: date
    date_to: date
    keywords: tuple[str, ...] = ()
    themes: tuple[str, ...] = ()
    source_profile: str | None = None
    sources: tuple[str, ...] = ()
    language_filters: tuple[str, ...] = ("en",)
    include_urls: tuple[str, ...] = ()
    exclude_urls: tuple[str, ...] = ()


@dataclass(slots=True)
class SourceRequest:
    source_name: str
    query: EventQuery
    params: JsonDict = field(default_factory=dict)


@dataclass(slots=True)
class RemoteDocument:
    source_name: str
    external_id: str
    title: str
    canonical_url: str
    published_at: datetime | None = None
    language: str = "en"
    metadata: JsonDict = field(default_factory=dict)


@dataclass(slots=True)
class DownloadedDocument:
    source_name: str
    remote_document: RemoteDocument
    retrieved_url: str
    content_type: str = "text/html"
    body_text: str | None = None
    body_bytes: bytes | None = None
    local_path: Path | None = None
    metadata: JsonDict = field(default_factory=dict)


@dataclass(slots=True)
class ExtractedContent:
    text: str
    extraction_method: str
    attachments: tuple[str, ...] = ()
    metadata: JsonDict = field(default_factory=dict)


@dataclass(slots=True)
class ExtractedDocument:
    downloaded_document: DownloadedDocument
    content: ExtractedContent


@dataclass(slots=True)
class DocumentRecord:
    document_id: str
    external_id: str
    source_connector: str
    publisher: str
    trust_tier: str
    canonical_url: str
    retrieved_url: str
    title: str
    published_at: datetime | None
    language: str
    content_type: str
    country_codes: tuple[str, ...]
    event_label: str
    tags: tuple[str, ...] = ()
    raw_text: str = ""
    clean_text: str = ""
    attachments: tuple[str, ...] = ()
    content_hash: str = ""
    title_hash: str = ""
    license_or_terms_hint: str | None = None
    retrieval_metadata: JsonDict = field(default_factory=dict)


@dataclass(slots=True)
class ParagraphRecord:
    paragraph_id: str
    document_id: str
    paragraph_index: int
    text: str
    start_char: int
    end_char: int
    source_connector: str
    publisher: str
    canonical_url: str
    retrieved_url: str
    title: str
    published_at: datetime | None
    language: str
    country_codes: tuple[str, ...]
    event_label: str
    tags: tuple[str, ...] = ()
    trust_tier: str = ""
    attachments: tuple[str, ...] = ()
    metadata: JsonDict = field(default_factory=dict)


@dataclass(slots=True)
class ClusterRecord:
    cluster_id: str
    cluster_label: str
    event_label: str
    paragraph_count: int
    document_count: int
    paragraph_ids: tuple[str, ...] = ()
    document_ids: tuple[str, ...] = ()
    top_terms: tuple[str, ...] = ()
    source_mix: JsonDict = field(default_factory=dict)
    representative_paragraph_id: str | None = None
    metadata: JsonDict = field(default_factory=dict)


@dataclass(slots=True)
class ClusterAssignment:
    cluster_id: str
    paragraph_id: str
    document_id: str
    paragraph_index: int
    score: float


@dataclass(slots=True)
class QuestionRecord:
    question_id: str
    cluster_id: str
    event_label: str
    question: str
    question_type: str
    priority: int
    metadata: JsonDict = field(default_factory=dict)


@dataclass(slots=True)
class SdgRecord:
    sdg_key: str
    sdg_number: int
    sdg_label: str
    event_label: str
    question_count: int
    answer_count: int
    question_ids: tuple[str, ...] = ()
    answer_ids: tuple[str, ...] = ()
    metadata: JsonDict = field(default_factory=dict)


@dataclass(slots=True)
class AnswerRecord:
    answer_id: str
    question_id: str
    cluster_id: str
    event_label: str
    answer: str
    citation_paragraph_ids: tuple[str, ...] = ()
    confidence: float = 0.0
    metadata: JsonDict = field(default_factory=dict)


@dataclass(slots=True)
class SummaryRecord:
    summary_id: str
    event_label: str
    scope: str
    scope_id: str
    title: str
    summary: str
    citation_paragraph_ids: tuple[str, ...] = ()
    metadata: JsonDict = field(default_factory=dict)


@dataclass(slots=True)
class DuplicateEdge:
    left_document_id: str
    right_document_id: str
    reason: str
    score: float
    preferred_document_id: str
