# Deployment

## Current Deployment Model

The modernized pipeline is best treated as a batch job rather than a request/response web service. A typical production run:

1. resolves a configured event window,
2. ingests source documents,
3. materializes processed artifacts,
4. writes JSON/Markdown reports,
5. publishes or archives outputs downstream.

## Recommended Deployment Patterns

### 1. Local or Analyst Workstation

Good for:

- exploratory runs,
- prompt iteration,
- fixture-based regression checks,
- one-off report generation.

Use:

- a Python virtual environment,
- repo-root `.env`,
- local filesystem artifact storage.

### 2. Scheduled Batch Runner

Good for:

- recurring country/event updates,
- nightly or weekly report refreshes,
- organization-internal automation.

Recommended components:

- cron, systemd timer, or CI scheduler,
- persistent volume for `data/` and `artifacts/`,
- centralized secrets injection,
- external log collection.

### 3. Containerized Worker

Good for:

- reproducible runtime environments,
- orchestrated deployments,
- multi-tenant execution queues.

Recommended container concerns:

- mount persistent storage for raw cache and outputs,
- inject secrets via environment variables,
- preinstall the optional clustering stack,
- warm model/download caches where practical.

## Secrets

Do not commit `.env`.

Required production secrets typically include:

- `OPENAI_API_KEY`
- `RELIEFWEB_APPNAME`

Optional:

- `OPENAI_BASE_URL`

## Persistence

At minimum, persist:

- `data/raw/`
- `data/processed/`
- `artifacts/reports/`
- `artifacts/ingestion/`

This makes runs auditable and allows replay/debugging of cached source responses.

## Operational Checklist

Before a scheduled production run, verify:

- source credentials and app names are valid,
- model endpoints support the configured model IDs,
- disk space is sufficient for raw cache growth,
- output directories are writable,
- network egress to ReliefWeb and model endpoints is available.

## Suggested First Production Hardening Steps

- add structured application logging,
- emit run duration and token-usage metrics,
- add regression fixtures for representative events,
- add container packaging,
- add CI validation for config examples and smoke tests,
- add retention policy for raw caches and generated artifacts.
