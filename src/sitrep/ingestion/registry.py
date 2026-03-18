from __future__ import annotations

from collections.abc import Callable

from sitrep.ingestion.base import SourceConnector
from sitrep.ingestion.fixture_client import FixtureManifestConnector
from sitrep.ingestion.reliefweb_client import ReliefWebConnector
from sitrep.ingestion.rss_client import RSSConnector
from sitrep.ingestion.web_source_client import GenericHtmlConnector
from sitrep.settings import AppSettings, SourceSettings
from sitrep.types import SourceKind

ConnectorFactory = Callable[[str, SourceSettings, AppSettings], SourceConnector]


class SourceRegistry:
    def __init__(self) -> None:
        self._factories: dict[SourceKind, ConnectorFactory] = {}

    def register(self, kind: SourceKind, factory: ConnectorFactory) -> None:
        self._factories[kind] = factory

    def create_connector(
        self,
        source_name: str,
        settings: SourceSettings,
        app_settings: AppSettings,
    ) -> SourceConnector:
        try:
            factory = self._factories[settings.kind]
        except KeyError as error:
            raise KeyError(f"No connector registered for source kind: {settings.kind.value}") from error
        return factory(source_name, settings, app_settings)

    def build_enabled_connectors(
        self,
        app_settings: AppSettings,
        selected_source_names: tuple[str, ...],
    ) -> list[SourceConnector]:
        connectors: list[SourceConnector] = []
        for source_name in selected_source_names:
            settings = app_settings.sources.get(source_name)
            if settings is None or not settings.enabled:
                continue
            connectors.append(self.create_connector(source_name, settings, app_settings))
        return connectors



def build_default_registry() -> SourceRegistry:
    registry = SourceRegistry()
    registry.register(
        SourceKind.RELIEFWEB_API,
        lambda source_name, settings, app_settings: ReliefWebConnector(
            source_name,
            settings,
            app_settings.storage.raw_dir,
            cache_enabled=app_settings.run.cache_enabled,
            offline_mode=app_settings.run.offline_mode,
        ),
    )
    registry.register(
        SourceKind.RSS_HTML,
        lambda source_name, settings, app_settings: RSSConnector(
            source_name,
            settings,
            app_settings.storage.raw_dir,
            cache_enabled=app_settings.run.cache_enabled,
            offline_mode=app_settings.run.offline_mode,
        ),
    )
    registry.register(
        SourceKind.GENERIC_HTML,
        lambda source_name, settings, app_settings: GenericHtmlConnector(
            source_name,
            settings,
            app_settings.storage.raw_dir,
            cache_enabled=app_settings.run.cache_enabled,
            offline_mode=app_settings.run.offline_mode,
        ),
    )
    registry.register(
        SourceKind.FIXTURE_MANIFEST,
        lambda source_name, settings, app_settings: FixtureManifestConnector(
            source_name,
            settings,
            app_settings.storage.raw_dir,
            cache_enabled=app_settings.run.cache_enabled,
            offline_mode=app_settings.run.offline_mode,
        ),
    )
    return registry
