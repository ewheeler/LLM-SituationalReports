# A Large-Language-Model Framework for Automated Humanitarian Situation Reporting

This repository contains the original notebook-based implementation for the paper *"A Large-Language-Model Framework for Automated Humanitarian Situation Reporting"* and an in-progress Python package rewrite under `src/sitrep/` for a more maintainable, deployment-friendly pipeline.

**Interactive report viewer:**  
https://idecost.github.io/LLM-SituationalReports/Viewer/viewer_v2.html

![Pipeline Overview](./Images/pipeline.png)

## Status

The repository now has two parallel implementations:

- `Codes/`: the original research notebooks and legacy artifact paths used in the paper.
- `src/sitrep/`: the new Python package and CLI for a production-style pipeline.

The modernized pipeline supports:

- multi-source ingestion with first-class ReliefWeb support,
- cached and offline source replay,
- paragraph-level provenance,
- notebook-style `UMAP + HDBSCAN` clustering,
- configurable question, answer, SDG, summary, and report stages,
- OpenAI-backed structured generation,
- cluster-organized and SDG-organized report outputs,
- report post-processing with citation renumbering and context recovery.

## Documentation

- `docs/installation.md`: environment setup, dependencies, and local development install.
- `docs/configuration.md`: `.env`, JSON/YAML config files, model settings, source settings, and runtime overrides.
- `docs/deployment.md`: recommended deployment patterns, operational checklist, and production considerations.

## Quick Start

### 1. Create an environment

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

For the full notebook-style clustering stack, also install:

```bash
python -m pip install -e '.[advanced-clustering]'
```

You can also install from `requirements.txt`, but the editable install is the best fit for active development.

### 2. Configure secrets and runtime settings

Copy the checked-in environment template:

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

The CLI automatically loads the repo-root `.env` file before resolving settings.

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

## Modernized Pipeline Overview

The new package lives under `src/sitrep/` and is organized as follows:

- `src/sitrep/settings.py`: typed application, source, storage, and generation settings.
- `src/sitrep/cli.py`: command-line interface.
- `src/sitrep/ingestion/`: source connectors, request builders, attachment parsing, and ingestion DAG nodes.
- `src/sitrep/preprocessing/`: paragraph extraction and provenance preservation.
- `src/sitrep/clustering/`: baseline and notebook-style clustering flows.
- `src/sitrep/questions/`: baseline and notebook-style multi-prompt question generation.
- `src/sitrep/answers/`: retrieval-backed answer synthesis.
- `src/sitrep/sdg/`: SDG classification and grouping.
- `src/sitrep/summaries/`: cluster, SDG, and executive summary generation.
- `src/sitrep/reports/`: report assembly, markdown rendering, citation post-processing, and context maps.
- `src/sitrep/llm/`: prompt registry, JSON schemas, and OpenAI Responses API integration.
- `src/sitrep/orchestration/`: Hamilton run profiles and driver wiring.

## Input Sources

The rewrite is no longer limited to local files. It supports:

- `reliefweb`: primary live source for humanitarian reports.
- `fixtures`: local manifest-backed corpora for offline development and regression tests.
- `rss` / `generic_html` patterns used by secondary connectors in the scaffold.

ReliefWeb raw request/response payloads are cached under `data/raw/reliefweb/`.

## Outputs

The modernized pipeline writes runtime outputs to:

- `data/processed/paragraphs/`
- `data/processed/clusters/`
- `data/processed/questions/`
- `data/processed/sdg_questions/`
- `data/processed/answers/`
- `data/processed/summaries/`
- `artifacts/reports/`
- `artifacts/ingestion/`

Generated reports are available in both JSON and Markdown forms, with cluster-organized and SDG-organized variants.

## Model-Backed Generation

The `generation` config block lets you switch stages independently between:

- `provider: baseline`
- `provider: openai`

Current defaults target concrete live model IDs:

- `gpt-5.4`
- `gpt-5.4-mini`

Cluster and executive summary stages now scale `max_output_tokens` from prompt size using configurable ratios, buffers, and caps to reduce structured-output truncation on larger evidence bundles.

## Deployment Notes

This repository does not yet include a first-party Dockerfile or infrastructure-as-code deployment package. The current recommendation is to deploy the CLI as a scheduled batch job or worker process. See `docs/deployment.md` for:

- local workstation deployment,
- cron / scheduler deployment,
- containerization guidance,
- secret management,
- cache and artifact persistence,
- production hardening recommendations.

## Repository Structure

```text
├── src/sitrep/         # Modern Python package rewrite
├── config/             # Example JSON/YAML settings for the new pipeline
├── docs/               # Installation, configuration, and deployment docs
├── Codes/              # Original notebook implementation
├── Results/            # Legacy intermediate artifacts used by the notebook workflow
├── Results_evaluation/ # Evaluation data
└── Viewer/             # Interactive report viewer
```

## Legacy Notebook Workflow

The original paper workflow remains in `Codes/` and still documents the research pipeline stages:

1. source selection,
2. cleaning and clustering,
3. question generation,
4. answer extraction,
5. cluster and SDG summaries,
6. executive summary generation,
7. report assembly and visualization.

Those notebooks continue to be useful as a methodological reference while the new `src/sitrep/` implementation is hardened for repeatable runs, testing, and deployment.
