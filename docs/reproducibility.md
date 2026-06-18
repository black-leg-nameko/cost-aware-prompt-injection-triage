# Reproducibility Notes

## Artifact Modes

This repository supports two modes:

- Aggregate check: use the sanitized JSON files under `cg_cascade/results_public/` to regenerate the report.
- Full reproduction: rerun data download, classifiers, embeddings, LLM judge calls, and derived analyses.
- NP calibration: run `cg_cascade/run_np_calibrated_triage.py` after router training and PromptShield judge checkpoints exist.
- Adaptive stress testing: run `cg_cascade/run_adaptive_padding_attack.py`
  after the main LLM judge outputs exist. Then run
  `cg_cascade/run_adaptive_padding_e2e.py`,
  `cg_cascade/run_sparse_mix_audit_e2e.py`,
  `cg_cascade/run_quarantine_session_e2e.py`, and
  `cg_cascade/run_sparse_adaptive_defense_experiments.py` for the
  end-to-end audit, quarantine, density-gate, and padding-variant analyses.
  `cg_cascade/run_probabilistic_audit_simulation.py` gives the
  routing-level audit exposure estimate.

The aggregate check is safe for publication. Full reproduction creates private files that are ignored by git.

## Expected Local Artifacts

Full reproduction creates:

- `cg_cascade/data/datasets/*.parquet`
- `cg_cascade/embeddings/cache/*.npy`
- `cg_cascade/models/saved/*.pkl`
- `cg_cascade/results/*.json` with per-sample `records`
- `cg_cascade/results/*.md` generated from private runs

These files must not be committed.

## External Services

Embedding and LLM judge experiments use an OpenAI-compatible API. Set `OPENAI_API_KEY` in the environment and optionally `OPENAI_BASE_URL`.

Token costs depend on provider pricing and prompt lengths. The paper distinguishes LLM call reduction from token-cost estimates; the included aggregate results preserve both where measured.
