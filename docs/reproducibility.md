# Reproducibility Notes

## Artifact Modes

This repository supports two modes:

- Aggregate check: use the sanitized JSON files under `cg_cascade/results_public/` to regenerate the report.
- Full reproduction: rerun data download, classifiers, embeddings, LLM judge calls, and derived analyses.
- Adaptive stress testing: run `cg_cascade/run_iwsec_adaptive_padding_attack.py` after the main LLM judge outputs exist.

The aggregate check is safe for publication. Full reproduction creates private files that are ignored by git.

## Expected Local Artifacts

Full reproduction creates:

- `cg_cascade/data/datasets/*.parquet`
- `cg_cascade/embeddings/cache/*.npy`
- `cg_cascade/models/saved/*.pkl`
- `cg_cascade/results/*.json` with per-sample `records`

These files must not be committed.

## External Services

Embedding and LLM judge experiments use an OpenAI-compatible API. Set `OPENAI_API_KEY` in the environment and optionally `OPENAI_BASE_URL`.

Token costs depend on provider pricing and prompt lengths. The paper distinguishes LLM call reduction from token-cost estimates; the included aggregate results preserve both where measured.
