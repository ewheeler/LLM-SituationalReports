# Installation

## Requirements

- Python `3.10+`
- `git`
- network access for live source retrieval and model-backed generation
- a valid `OPENAI_API_KEY` for model-backed stages
- an approved `RELIEFWEB_APPNAME` for live ReliefWeb API access

## Recommended Local Setup

```bash
git clone git@github.com:ewheeler/LLM-SituationalReports.git
cd LLM-SituationalReports
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

## Optional Extras

Install the notebook-style clustering dependencies:

```bash
python -m pip install -e '.[advanced-clustering]'
```

Install from the pinned requirements file instead:

```bash
python -m pip install -r requirements.txt
```

## Environment File

Copy the template:

```bash
cp .env.example .env
```

Populate at least:

```env
OPENAI_API_KEY=...
RELIEFWEB_APPNAME=...
```

If you are using a compatible non-default endpoint, also set:

```env
OPENAI_BASE_URL=...
```

## Smoke Tests

Verify the package and config resolution:

```bash
PYTHONPATH=src python -m sitrep.cli show-config
```

Verify the CLI can resolve a run payload:

```bash
PYTHONPATH=src python -m sitrep.cli ingest \
  --event "Sudan humanitarian situation" \
  --country SDN \
  --date-from 2024-08-19 \
  --date-to 2024-08-25 \
  --sources reliefweb \
  --dry-run
```
