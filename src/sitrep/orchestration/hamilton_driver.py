from __future__ import annotations

from importlib import import_module
from typing import Any

from sitrep.orchestration.run_profiles import RunProfile, get_run_profile
from sitrep.settings import AppSettings


class HamiltonUnavailableError(RuntimeError):
    pass



def build_driver(profile: RunProfile | None = None):
    selected_profile = profile or get_run_profile()
    try:
        from hamilton import driver
    except ModuleNotFoundError as error:
        raise HamiltonUnavailableError(
            "sf-hamilton is not installed. Install it to execute the ingestion DAG."
        ) from error

    modules = [import_module(module_path) for module_path in selected_profile.module_paths]
    if hasattr(driver, "Builder"):
        builder = driver.Builder()
        for module in modules:
            builder = builder.with_modules(module)
        return builder.build()

    from hamilton import base

    adapter = base.SimplePythonGraphAdapter(base.DictResult())
    return driver.Driver({}, *modules, adapter=adapter)



def execute_profile(
    app_settings: AppSettings,
    event_name: str,
    country_codes: tuple[str, ...],
    date_from,
    date_to,
    keywords: tuple[str, ...] = (),
    themes: tuple[str, ...] = (),
    source_profile: str | None = None,
    sources: tuple[str, ...] = (),
    include_urls: tuple[str, ...] = (),
    exclude_urls: tuple[str, ...] = (),
    profile_name: str = "ingest_only",
) -> dict[str, Any]:
    profile = get_run_profile(profile_name)
    driver = build_driver(profile)
    inputs = {
        "app_settings": app_settings,
        "event_name": event_name,
        "country_codes": country_codes,
        "date_from": date_from,
        "date_to": date_to,
        "keywords": keywords,
        "themes": themes,
        "source_profile": source_profile,
        "sources": sources,
        "include_urls": include_urls,
        "exclude_urls": exclude_urls,
    }
    return driver.execute(final_vars=list(profile.outputs), inputs=inputs)
