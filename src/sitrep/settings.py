from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping

try:
    import tomllib
except ModuleNotFoundError:
    try:
        import tomli as tomllib  # type: ignore
    except ModuleNotFoundError:
        tomllib = None

from sitrep.types import SourceKind, TrustTier

try:
    import yaml  # type: ignore
except ModuleNotFoundError:
    yaml = None

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    load_dotenv = None

_DOTENV_LOADED = False


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_repo_dotenv(*, override: bool = False) -> None:
    global _DOTENV_LOADED
    if _DOTENV_LOADED:
        return
    if load_dotenv is None:
        _DOTENV_LOADED = True
        return
    dotenv_path = _repo_root() / ".env"
    if dotenv_path.exists():
        load_dotenv(dotenv_path=dotenv_path, override=override)
    _DOTENV_LOADED = True


@dataclass(slots=True)
class ModelSettings:
    synthesis: str = "gpt-5.4"
    reasoning: str = "gpt-5.4-mini"
    extraction: str = "gpt-5.4-mini"
    embeddings: str = "text-embedding-3-large"


@dataclass(slots=True)
class RetrievalSettings:
    top_k: int = 4
    lexical_weight: float = 1.0
    title_weight: float = 0.3
    trust_weight: float = 0.2
    recency_weight: float = 0.1


@dataclass(slots=True)
class ClusteringSettings:
    provider: str = "baseline"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    random_search_evaluations: int = 16
    min_clusters: int = 5
    n_neighbors: tuple[int, ...] = (3, 4, 5, 6, 8, 10, 12)
    min_cluster_size: tuple[int, ...] = (3, 4, 5, 6, 8, 10)
    min_samples: tuple[int, ...] = (1, 2, 3, 4, 5, 6)
    cluster_selection_epsilon: tuple[float, ...] = (0.0, 0.01, 0.03, 0.05, 0.1)
    umap_components: int = 10
    umap_metric: str = "cosine"
    random_state: int = 101
    noise_penalty: float = 0.25


@dataclass(slots=True)
class GenerationStageSettings:
    provider: str = "baseline"
    model: str | None = None
    prompt_id: str | None = None
    temperature: float = 0.2
    max_output_tokens: int = 1200
    scale_output_with_input: bool = False
    output_token_estimate_ratio: float = 0.0
    output_token_buffer: int = 0
    max_output_tokens_cap: int | None = None
    fanout_prompt_ids: tuple[str, ...] = ()
    max_questions_per_cluster: int = 6
    within_cluster_dedupe_threshold: float = 0.22
    across_cluster_dedupe_threshold: float = 0.55


@dataclass(slots=True)
class SdgSettings:
    provider: str = "baseline"
    model: str | None = "gpt-5.4-mini"
    prompt_id: str | None = "sdg.classify.v1"
    max_labels_per_question: int = 3
    min_score: float = 0.18


@dataclass(slots=True)
class GenerationSettings:
    questions: GenerationStageSettings = field(
        default_factory=lambda: GenerationStageSettings(
            provider="baseline",
            model="gpt-5.4-mini",
            prompt_id="questions.cluster.v1",
            temperature=0.3,
            max_output_tokens=900,
            fanout_prompt_ids=(
                "questions.cluster.v1",
                "questions.cluster.v2",
                "questions.cluster.v3",
            ),
            max_questions_per_cluster=6,
            within_cluster_dedupe_threshold=0.22,
            across_cluster_dedupe_threshold=0.55,
        )
    )
    answers: GenerationStageSettings = field(
        default_factory=lambda: GenerationStageSettings(
            provider="baseline",
            model="gpt-5.4",
            prompt_id="answers.grounded.v1",
            temperature=0.2,
            max_output_tokens=1200,
        )
    )
    cluster_summaries: GenerationStageSettings = field(
        default_factory=lambda: GenerationStageSettings(
            provider="baseline",
            model="gpt-5.4",
            prompt_id="summaries.cluster.v2",
            temperature=0.2,
            max_output_tokens=1200,
            scale_output_with_input=True,
            output_token_estimate_ratio=0.6,
            output_token_buffer=200,
            max_output_tokens_cap=3200,
        )
    )
    executive_summary: GenerationStageSettings = field(
        default_factory=lambda: GenerationStageSettings(
            provider="baseline",
            model="gpt-5.4",
            prompt_id="summaries.executive.v2",
            temperature=0.2,
            max_output_tokens=1000,
            scale_output_with_input=True,
            output_token_estimate_ratio=0.8,
            output_token_buffer=300,
            max_output_tokens_cap=2600,
        )
    )


@dataclass(slots=True)
class StorageSettings:
    project_root: Path = Path(".")
    raw_dir: Path = Path("data/raw")
    normalized_documents_dir: Path = Path("data/normalized/documents")
    documents_manifest_path: Path = Path("data/normalized/documents_manifest.parquet")
    duplicates_path: Path = Path("data/normalized/duplicates.parquet")
    paragraphs_dir: Path = Path("data/processed/paragraphs")
    clusters_dir: Path = Path("data/processed/clusters")
    questions_dir: Path = Path("data/processed/questions")
    sdg_questions_dir: Path = Path("data/processed/sdg_questions")
    answers_dir: Path = Path("data/processed/answers")
    summaries_dir: Path = Path("data/processed/summaries")
    reports_dir: Path = Path("artifacts/reports")
    artifacts_dir: Path = Path("artifacts")


@dataclass(slots=True)
class SourceSettings:
    enabled: bool
    kind: SourceKind
    trust_tier: TrustTier
    priority: int
    languages: tuple[str, ...] = ("en",)
    base_url: str | None = None
    feed_url: str | None = None
    fixture_manifest_path: str | None = None
    extraction_mode: str = "auto"
    timeout_seconds: int = 30
    max_requests_per_minute: int = 30
    appname: str | None = None
    user_agent: str = "sitrep/0.1"
    page_size: int = 25
    max_pages: int = 2


@dataclass(slots=True)
class RunSettings:
    cache_enabled: bool = True
    offline_mode: bool = False
    default_source_profile: str = "humanitarian_core"


@dataclass(slots=True)
class AppSettings:
    models: ModelSettings = field(default_factory=ModelSettings)
    retrieval: RetrievalSettings = field(default_factory=RetrievalSettings)
    clustering: ClusteringSettings = field(default_factory=ClusteringSettings)
    generation: GenerationSettings = field(default_factory=GenerationSettings)
    sdg: SdgSettings = field(default_factory=SdgSettings)
    storage: StorageSettings = field(default_factory=StorageSettings)
    run: RunSettings = field(default_factory=RunSettings)
    sources: dict[str, SourceSettings] = field(default_factory=dict)
    source_profiles: dict[str, tuple[str, ...]] = field(default_factory=dict)
    openai_api_key: str | None = None
    openai_base_url: str | None = None

    @classmethod
    def default(cls) -> "AppSettings":
        load_repo_dotenv()
        return cls(
            sources={
                "reliefweb": SourceSettings(
                    enabled=True,
                    kind=SourceKind.RELIEFWEB_API,
                    trust_tier=TrustTier.HUMANITARIAN_PRIMARY,
                    priority=100,
                    base_url="https://api.reliefweb.int/v2/reports",
                    appname=os.getenv("RELIEFWEB_APPNAME"),
                    user_agent="sitrep/0.1 (+https://github.com/idecost/LLM-SituationalReports)",
                    page_size=25,
                    max_pages=2,
                ),
                "preventionweb": SourceSettings(
                    enabled=True,
                    kind=SourceKind.RSS_HTML,
                    trust_tier=TrustTier.HUMANITARIAN_PRIMARY,
                    priority=90,
                    base_url="https://www.preventionweb.net",
                    feed_url=os.getenv("PREVENTIONWEB_FEED_URL"),
                ),
                "the_new_humanitarian": SourceSettings(
                    enabled=True,
                    kind=SourceKind.RSS_HTML,
                    trust_tier=TrustTier.HUMANITARIAN_PRIMARY,
                    priority=85,
                    base_url="https://www.thenewhumanitarian.org",
                    feed_url=os.getenv("THE_NEW_HUMANITARIAN_FEED_URL"),
                ),
                "bbc": SourceSettings(
                    enabled=False,
                    kind=SourceKind.GENERIC_HTML,
                    trust_tier=TrustTier.MAJOR_NEWS,
                    priority=60,
                    base_url="https://www.bbc.com",
                    feed_url=os.getenv("BBC_FEED_URL"),
                ),
                "fixtures": SourceSettings(
                    enabled=False,
                    kind=SourceKind.FIXTURE_MANIFEST,
                    trust_tier=TrustTier.CURATED_WEB,
                    priority=70,
                    fixture_manifest_path=os.getenv("SITREP_FIXTURE_MANIFEST"),
                ),
            },
            source_profiles={
                "humanitarian_core": (
                    "reliefweb",
                    "preventionweb",
                    "the_new_humanitarian",
                ),
                "humanitarian_plus_news": (
                    "reliefweb",
                    "preventionweb",
                    "the_new_humanitarian",
                    "bbc",
                ),
                "local_fixture": ("fixtures",),
            },
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_base_url=os.getenv("OPENAI_BASE_URL"),
        )

    def selected_sources(
        self,
        explicit_sources: tuple[str, ...] = (),
        source_profile: str | None = None,
    ) -> tuple[str, ...]:
        if explicit_sources:
            return tuple(source for source in explicit_sources if source in self.sources)
        profile_name = source_profile or self.run.default_source_profile
        return self.source_profiles.get(profile_name, ())

    def with_runtime_overrides(
        self,
        *,
        offline_mode: bool | None = None,
        cache_enabled: bool | None = None,
        reliefweb_appname: str | None = None,
        fixture_manifest_path: str | None = None,
    ) -> "AppSettings":
        updated = replace(
            self,
            retrieval=replace(self.retrieval),
            clustering=replace(self.clustering),
            generation=replace(
                self.generation,
                questions=replace(self.generation.questions),
                answers=replace(self.generation.answers),
                cluster_summaries=replace(self.generation.cluster_summaries),
                executive_summary=replace(self.generation.executive_summary),
            ),
            sdg=replace(self.sdg),
            run=replace(self.run),
        )
        if offline_mode is not None:
            updated.run.offline_mode = offline_mode
        if cache_enabled is not None:
            updated.run.cache_enabled = cache_enabled
        updated.sources = {
            name: replace(source_settings)
            for name, source_settings in self.sources.items()
        }
        if reliefweb_appname is not None and "reliefweb" in updated.sources:
            updated.sources["reliefweb"].appname = reliefweb_appname
        if fixture_manifest_path is not None and "fixtures" in updated.sources:
            updated.sources["fixtures"].fixture_manifest_path = fixture_manifest_path
            updated.sources["fixtures"].enabled = True
        return updated

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["models"] = asdict(self.models)
        payload["retrieval"] = asdict(self.retrieval)
        payload["clustering"] = {
            **asdict(self.clustering),
            "n_neighbors": list(self.clustering.n_neighbors),
            "min_cluster_size": list(self.clustering.min_cluster_size),
            "min_samples": list(self.clustering.min_samples),
            "cluster_selection_epsilon": list(self.clustering.cluster_selection_epsilon),
        }
        payload["generation"] = {
            "questions": {
                **asdict(self.generation.questions),
                "fanout_prompt_ids": list(self.generation.questions.fanout_prompt_ids),
            },
            "answers": {
                **asdict(self.generation.answers),
                "fanout_prompt_ids": list(self.generation.answers.fanout_prompt_ids),
            },
            "cluster_summaries": {
                **asdict(self.generation.cluster_summaries),
                "fanout_prompt_ids": list(self.generation.cluster_summaries.fanout_prompt_ids),
            },
            "executive_summary": {
                **asdict(self.generation.executive_summary),
                "fanout_prompt_ids": list(self.generation.executive_summary.fanout_prompt_ids),
            },
        }
        payload["sdg"] = asdict(self.sdg)
        payload["storage"] = {key: str(value) for key, value in asdict(self.storage).items()}
        payload["run"] = asdict(self.run)
        payload["sources"] = {
            name: {
                **asdict(config),
                "kind": config.kind.value,
                "trust_tier": config.trust_tier.value,
                "languages": list(config.languages),
            }
            for name, config in self.sources.items()
        }
        payload["source_profiles"] = {
            name: list(sources)
            for name, sources in self.source_profiles.items()
        }
        return payload

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "AppSettings":
        default_settings = cls.default().to_dict()
        merged = _deep_merge(default_settings, dict(data))
        storage = StorageSettings(**{key: Path(value) for key, value in merged["storage"].items()})
        sources = {
            name: SourceSettings(
                enabled=payload["enabled"],
                kind=SourceKind(payload["kind"]),
                trust_tier=TrustTier(payload["trust_tier"]),
                priority=payload["priority"],
                languages=tuple(payload.get("languages", ("en",))),
                base_url=payload.get("base_url"),
                feed_url=payload.get("feed_url"),
                fixture_manifest_path=payload.get("fixture_manifest_path"),
                extraction_mode=payload.get("extraction_mode", "auto"),
                timeout_seconds=payload.get("timeout_seconds", 30),
                max_requests_per_minute=payload.get("max_requests_per_minute", 30),
                appname=payload.get("appname"),
                user_agent=payload.get("user_agent", "sitrep/0.1"),
                page_size=payload.get("page_size", 25),
                max_pages=payload.get("max_pages", 2),
            )
            for name, payload in merged["sources"].items()
        }
        clustering_payload = merged.get("clustering", {})
        generation_payload = merged.get("generation", {})

        def stage_settings(name: str) -> GenerationStageSettings:
            payload = generation_payload.get(name, {})
            return GenerationStageSettings(
                provider=payload.get("provider", "baseline"),
                model=payload.get("model"),
                prompt_id=payload.get("prompt_id"),
                temperature=float(payload.get("temperature", 0.2)),
                max_output_tokens=int(payload.get("max_output_tokens", 1200)),
                scale_output_with_input=bool(payload.get("scale_output_with_input", False)),
                output_token_estimate_ratio=float(payload.get("output_token_estimate_ratio", 0.0)),
                output_token_buffer=int(payload.get("output_token_buffer", 0)),
                max_output_tokens_cap=(
                    int(payload["max_output_tokens_cap"])
                    if payload.get("max_output_tokens_cap") is not None
                    else None
                ),
                fanout_prompt_ids=tuple(payload.get("fanout_prompt_ids", ())),
                max_questions_per_cluster=int(payload.get("max_questions_per_cluster", 6)),
                within_cluster_dedupe_threshold=float(payload.get("within_cluster_dedupe_threshold", 0.22)),
                across_cluster_dedupe_threshold=float(payload.get("across_cluster_dedupe_threshold", 0.55)),
            )

        sdg_payload = merged.get("sdg", {})
        return cls(
            models=ModelSettings(**merged["models"]),
            retrieval=RetrievalSettings(**merged.get("retrieval", {})),
            clustering=ClusteringSettings(
                provider=clustering_payload.get("provider", "baseline"),
                embedding_model=clustering_payload.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2"),
                random_search_evaluations=int(clustering_payload.get("random_search_evaluations", 16)),
                min_clusters=int(clustering_payload.get("min_clusters", 5)),
                n_neighbors=tuple(int(value) for value in clustering_payload.get("n_neighbors", (3, 4, 5, 6, 8, 10, 12))),
                min_cluster_size=tuple(int(value) for value in clustering_payload.get("min_cluster_size", (3, 4, 5, 6, 8, 10))),
                min_samples=tuple(int(value) for value in clustering_payload.get("min_samples", (1, 2, 3, 4, 5, 6))),
                cluster_selection_epsilon=tuple(float(value) for value in clustering_payload.get("cluster_selection_epsilon", (0.0, 0.01, 0.03, 0.05, 0.1))),
                umap_components=int(clustering_payload.get("umap_components", 10)),
                umap_metric=clustering_payload.get("umap_metric", "cosine"),
                random_state=int(clustering_payload.get("random_state", 101)),
                noise_penalty=float(clustering_payload.get("noise_penalty", 0.25)),
            ),
            generation=GenerationSettings(
                questions=stage_settings("questions"),
                answers=stage_settings("answers"),
                cluster_summaries=stage_settings("cluster_summaries"),
                executive_summary=stage_settings("executive_summary"),
            ),
            sdg=SdgSettings(
                provider=sdg_payload.get("provider", "baseline"),
                model=sdg_payload.get("model", "gpt-5.4-mini"),
                prompt_id=sdg_payload.get("prompt_id", "sdg.classify.v1"),
                max_labels_per_question=int(sdg_payload.get("max_labels_per_question", 3)),
                min_score=float(sdg_payload.get("min_score", 0.18)),
            ),
            storage=storage,
            run=RunSettings(**merged["run"]),
            sources=sources,
            source_profiles={
                name: tuple(values)
                for name, values in merged["source_profiles"].items()
            },
            openai_api_key=merged.get("openai_api_key") or os.getenv("OPENAI_API_KEY"),
            openai_base_url=merged.get("openai_base_url") or os.getenv("OPENAI_BASE_URL"),
        )


def load_settings(config_path: str | Path | None = None) -> AppSettings:
    load_repo_dotenv()
    if config_path is None:
        return AppSettings.default()

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Settings file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".json":
        raw_data = json.loads(path.read_text())
    elif suffix == ".toml":
        if tomllib is None:
            raise RuntimeError("TOML settings require Python 3.11+ or the tomli package.")
        raw_data = tomllib.loads(path.read_text())
    elif suffix in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError("PyYAML is required to load YAML settings files.")
        raw_data = yaml.safe_load(path.read_text()) or {}
    else:
        raise ValueError(f"Unsupported settings format: {suffix}")

    return AppSettings.from_mapping(raw_data)



def _deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
            continue
        merged[key] = value
    return merged
