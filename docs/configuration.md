# Configuration

## Configuration Layers

The modern pipeline resolves configuration from three places, in this order:

1. built-in defaults in `src/sitrep/settings.py`
2. repo-root `.env`
3. JSON, YAML, or TOML config passed with `--config`

Runtime flags such as `--offline-mode`, `--disable-cache`, `--reliefweb-appname`, and `--fixture-manifest` override the resolved settings for a given run.

## Environment Variables

Supported environment-backed values include:

- `OPENAI_API_KEY`
- `OPENAI_BASE_URL`
- `RELIEFWEB_APPNAME`
- `BBC_FEED_URL`
- `SITREP_FIXTURE_MANIFEST`

## Example Config Files

The repository includes:

- `config/settings.example.json`
- `config/settings.example.yaml`

These show how to configure:

- model provider selection,
- clustering provider,
- source enablement,
- output-token scaling for summaries,
- SDG classification,
- storage paths,
- cache and offline behavior.

## Generation Settings

Each generation stage supports its own provider and model:

- `questions`
- `answers`
- `cluster_summaries`
- `executive_summary`

Summary stages additionally support dynamic output scaling:

- `scale_output_with_input`
- `output_token_estimate_ratio`
- `output_token_buffer`
- `max_output_tokens_cap`

This is useful for long evidence bundles where structured-output responses might otherwise truncate.

## Source Settings

Each source entry can define:

- `enabled`
- `kind`
- `trust_tier`
- `priority`
- `base_url`
- `feed_url`
- `fixture_manifest_path`
- `page_size`
- `max_pages`
- `appname`
- `user_agent`

## Storage Paths

Key output locations are configured under `storage` and default to:

- `data/raw`
- `data/processed/*`
- `artifacts/reports`
- `artifacts/ingestion`

For local development, the defaults are usually sufficient. For deployment, consider moving these paths to persistent volumes.
