# Security Review

## Release Boundary

The public boundary is code plus sanitized aggregate results. It intentionally excludes:

- API keys and dotenv files.
- Raw prompt corpora and downloaded dataset files.
- Embedding vectors and trained model artifacts.
- Per-sample LLM outputs, judge reasons, and qualitative examples.
- Paper drafts and local build outputs.

The probabilistic bypass-audit artifact is a routing-level exposure simulation
over sanitized aggregate Table 6 streams. It does not include fresh LLM judge
outputs for padded prompts or raw prompt text.

## Pre-Push Checklist

1. Confirm `.env` and credential files are absent.
2. Confirm no `*.parquet`, `*.npy`, `*.pkl`, or `*.joblib` files are tracked.
3. Confirm result JSONs do not contain `records`, `prompt_excerpt`, `examples`, or `top_examples`.
4. Confirm no local absolute paths or account names appear in tracked files.
5. Run `python tests/test_no_sensitive_files.py`.
6. Inspect `git status --short` before pushing.

## Design Note

The shift monitor is not an attack detector. It checks whether the router is still operating on a source distribution close enough to its validation distribution. When shift is detected, the system gives up bypass savings and falls back to LLM-only judging.
