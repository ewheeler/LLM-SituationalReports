# A Large-Language-Model Framework for Automated Humanitarian Situation Reporting

This repository contains the modernized `src/sitrep/` implementation of an automated humanitarian situation-report generation pipeline.

The current codebase is organized around a production-style Python package and CLI rather than the earlier notebook-oriented workflow. It supports typed configuration, multi-source ingestion, paragraph-level provenance, notebook-style clustering, model-backed generation, and report assembly with citation-aware post-processing.

## Capabilities

The modern pipeline supports:

- ReliefWeb-first live ingestion with cached raw payload storage,
- fixture-backed offline regression runs,
- paragraph splitting and provenance-preserving preprocessing,
- notebook-style `UMAP + HDBSCAN` clustering,
- configurable question, answer, SDG, summary, and report stages,
- OpenAI-backed structured generation,
- cluster-organized and SDG-organized report outputs,
- citation renumbering, context recovery, and summary context maps.

## Documentation

- `docs/installation.md`: environment setup and dependency installation.
- `docs/configuration.md`: `.env`, JSON/YAML config, runtime flags, and stage settings.
- `docs/deployment.md`: deployment patterns, persistence, secrets, and production hardening.

## Quick Start

### 1. Create an environment

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

For notebook-style clustering support:

```bash
python -m pip install -e '.[advanced-clustering]'
```

### 2. Configure runtime secrets

```bash
cp .env.example .env
```

At minimum, set:

- `OPENAI_API_KEY`
- `RELIEFWEB_APPNAME`

Optional:

- `OPENAI_BASE_URL`
- `BBC_FEED_URL`
- `SITREP_FIXTURE_MANIFEST`

The CLI automatically loads the repo-root `.env` file.

### 3. Inspect resolved configuration

```bash
PYTHONPATH=src python -m sitrep.cli show-config
```

### 4. Run a dry run

```bash
PYTHONPATH=src python -m sitrep.cli ingest \
  --event "Sudan humanitarian situation" \
  --country SDN \
  --date-from 2024-08-19 \
  --date-to 2024-08-25 \
  --sources reliefweb \
  --dry-run
```

### 5. Run a live end-to-end ingest

```bash
PYTHONPATH=src python -m sitrep.cli ingest \
  --config config/settings.example.json \
  --event "Sudan humanitarian situation" \
  --country SDN \
  --date-from 2024-08-19 \
  --date-to 2024-08-25 \
  --sources reliefweb
```

## Package Layout

```text
├── src/sitrep/         # Modern Python package
├── config/             # Example JSON/YAML settings
├── docs/               # Installation, configuration, deployment docs
├── requirements.txt    # Environment dependencies
└── pyproject.toml      # Packaging metadata
```

Key modules include:

- `src/sitrep/settings.py`: typed application and source settings.
- `src/sitrep/cli.py`: CLI entrypoint.
- `src/sitrep/ingestion/`: source connectors, request builders, and attachment parsing.
- `src/sitrep/preprocessing/`: paragraph extraction and normalization.
- `src/sitrep/clustering/`: baseline and notebook-style clustering.
- `src/sitrep/questions/`: baseline and notebook-style question generation.
- `src/sitrep/answers/`: retrieval-backed answer generation.
- `src/sitrep/sdg/`: SDG classification and grouping.
- `src/sitrep/summaries/`: cluster, SDG, and executive summary generation.
- `src/sitrep/reports/`: report assembly, markdown rendering, and citation post-processing.
- `src/sitrep/llm/`: prompts, schemas, and OpenAI Responses API integration.
- `src/sitrep/orchestration/`: Hamilton run profiles and driver wiring.

## Inputs and Outputs

### Sources

The current scaffold supports:

- `reliefweb`: primary live humanitarian source,
- `fixtures`: local manifest-backed corpora for offline development,
- RSS / generic HTML connector patterns for secondary ingestion.

ReliefWeb request and response payloads are cached under `data/raw/reliefweb/`.

### Outputs

The pipeline writes runtime artifacts to:

- `data/processed/paragraphs/`
- `data/processed/clusters/`
- `data/processed/questions/`
- `data/processed/sdg_questions/`
- `data/processed/answers/`
- `data/processed/summaries/`
- `artifacts/reports/`
- `artifacts/ingestion/`

Reports are emitted in JSON and Markdown, with both cluster-organized and SDG-organized variants.

## Model-Backed Generation

Each generation stage can run with:

- `provider: baseline`
- `provider: openai`

The current defaults target:

- `gpt-5.4`
- `gpt-5.4-mini`

Cluster and executive summary stages can scale `max_output_tokens` from prompt size using configurable ratios, buffers, and caps to reduce structured-output truncation.

## Deployment

This repository is currently best deployed as a batch pipeline rather than a web app. See `docs/deployment.md` for guidance on:

- workstation use,
- scheduled jobs,
- containerized workers,
- persistence and cache handling,
- secrets,
- production hardening.
