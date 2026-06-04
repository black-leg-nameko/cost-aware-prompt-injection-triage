# Security Policy

This artifact is for defensive prompt-injection detection research. It does not include secrets, raw prompt corpora, model checkpoints, embedding caches, or full LLM judge records.

## Reporting Issues

Please report accidental secret exposure, data leakage, unsafe release artifacts, or reproducibility problems through the repository issue tracker or the private reporting channel available to collaborators.

## Artifact Handling Rules

- Do not commit `.env`, API tokens, provider logs, downloaded datasets, embeddings, trained model files, or raw LLM records.
- Treat prompt corpora and per-sample judge reasons as private research artifacts unless the publication venue explicitly permits their redistribution.
- Run `python tests/test_no_sensitive_files.py` before committing or pushing.
- Sanitize full experiment outputs with `tools/sanitize_results.py` before sharing aggregate results.
