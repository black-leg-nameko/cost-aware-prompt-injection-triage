# Cost-Aware Prompt-Injection Triage

This repository contains the experiment code for a cost-aware direct prompt-injection detection study. The artifact evaluates local classical detectors, OpenAI-embedding classifiers, an LLM-as-a-judge detector, classifier-based bypass routing, and stream-level shift monitoring with fail-safe fallback.

The release is intentionally conservative: it includes experiment programs and sanitized aggregate results, but excludes raw prompts, downloaded datasets, embeddings, trained model artifacts, API logs, and environment files.

## What Is Included

- Dataset preparation scripts for the semantic direct prompt-injection benchmark suite.
- TF-IDF logistic-regression and OpenAI-embedding classifier baselines.
- LLM judge evaluation with checkpointing and router metrics.
- Same-subset classifier-vs-LLM comparison.
- Source-matched Neyman-Pearson/Clopper-Pearson threshold calibration.
- Shift-monitor fail-safe experiments using PSI, KS tests, and keyword-rate drift.
- Mixed low-rate OOD diagnostic probe.
- White-box benign-feature padding stress test against the TF-IDF router.
- Probabilistic bypass-audit exposure simulation for sparse adaptive streams.
- False-negative category audit with public-artifact mode that records counts only.
- Sanitized aggregate result JSONs and a generated experiment report.

## What Is Not Included

- `.env` files, API keys, tokens, or local credentials.
- Raw dataset files (`*.parquet`) and prompt texts.
- Embedding caches (`*.npy`) and chunk caches.
- Trained model files (`*.pkl`, `*.joblib`).
- Per-sample LLM judge records and raw qualitative examples.
- Paper PDFs, drafts, logs, or build artifacts.

## Repository Layout

```text
cg_cascade/
  prepare_iwsec_datasets.py
  run_iwsec_classical_experiments.py
  embed_iwsec_datasets.py
  run_iwsec_embedding_experiments.py
  evaluate_iwsec_ayub_public_rf.py
  run_iwsec_openai_judge.py
  evaluate_iwsec_subset_classifier_comparison.py
  run_iwsec_np_calibrated_triage.py
  run_iwsec_shift_failsafe_experiment.py
  run_iwsec_mixed_shift_probe.py
  run_iwsec_adaptive_padding_attack.py
  run_iwsec_probabilistic_audit_simulation.py
  plot_false_bypass_pareto.py
  analyze_iwsec_llm_false_negatives.py
  summarize_iwsec_reworked_results.py
  results_public/                # sanitized aggregate results only
docs/
tools/
tests/
```

## Setup

Use Python 3.11 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Set API credentials through the environment. Do not commit `.env` files.

```bash
export OPENAI_API_KEY="..."
```

The scripts also support `OPENAI_BASE_URL` for compatible API endpoints.

## Quick Aggregate Check

The included result files are sanitized aggregates under `cg_cascade/results_public/`. To regenerate the report tables from them:

```bash
python cg_cascade/summarize_iwsec_reworked_results.py
```

This does not require raw datasets or API access.

## Full Reproduction Workflow

The full workflow downloads public datasets and creates local artifacts that are intentionally ignored by git.

```bash
python cg_cascade/prepare_iwsec_datasets.py
python cg_cascade/run_iwsec_classical_experiments.py
python cg_cascade/embed_iwsec_datasets.py
python cg_cascade/run_iwsec_embedding_experiments.py
```

If evaluating the public Ayub et al. random-forest model, place the model file at:

```text
cg_cascade/models/saved/ayub_rf_openai.pkl
```

Then run:

```bash
python cg_cascade/evaluate_iwsec_ayub_public_rf.py
```

Run the LLM judge on the paper subsets and external streams:

```bash
python cg_cascade/run_iwsec_openai_judge.py \
  --dataset promptshield_test_balanced2000 \
  --dataset-path cg_cascade/data/datasets/promptshield_test.parquet \
  --output cg_cascade/results/iwsec_openai_promptshield_test_balanced2000.json \
  --sample-size 2000 \
  --balanced-sample

python cg_cascade/run_iwsec_openai_judge.py \
  --dataset promptshield_test_natural3000 \
  --dataset-path cg_cascade/data/datasets/promptshield_test.parquet \
  --output cg_cascade/results/iwsec_openai_promptshield_test_natural3000.json \
  --sample-size 3000

python cg_cascade/run_iwsec_openai_judge.py \
  --dataset deepset_semantic_all \
  --dataset-path cg_cascade/data/datasets/deepset_semantic_all.parquet \
  --output cg_cascade/results/iwsec_openai_deepset_semantic_all.json \
  --sample-size 0

python cg_cascade/run_iwsec_openai_judge.py \
  --dataset notinject_hard_negatives \
  --dataset-path cg_cascade/data/datasets/notinject_hard_negatives.parquet \
  --output cg_cascade/results/iwsec_openai_notinject_hard_negatives.json \
  --sample-size 0

python cg_cascade/run_iwsec_openai_judge.py \
  --dataset lakera_gandalf_attack_only \
  --dataset-path cg_cascade/data/datasets/lakera_gandalf_attack_only.parquet \
  --output cg_cascade/results/iwsec_openai_lakera_gandalf_attack_only.json \
  --sample-size 0
```

Then run the derived analyses:

```bash
python cg_cascade/evaluate_iwsec_subset_classifier_comparison.py
python cg_cascade/run_iwsec_np_calibrated_triage.py
python cg_cascade/run_iwsec_shift_failsafe_experiment.py
python cg_cascade/run_iwsec_mixed_shift_probe.py
python cg_cascade/run_iwsec_adaptive_padding_attack.py
python cg_cascade/run_iwsec_probabilistic_audit_simulation.py
python cg_cascade/analyze_iwsec_llm_false_negatives.py
python cg_cascade/summarize_iwsec_reworked_results.py --results-dir cg_cascade/results
```

By default, `prepare_iwsec_datasets.py` and `analyze_iwsec_llm_false_negatives.py` do not write prompt excerpts to public-facing JSON. Use `--include-examples` only inside a private audit environment.

## Sanitizing Results Before Sharing

After a full rerun, sanitize result artifacts before publication:

```bash
python tools/sanitize_results.py \
  --source-results cg_cascade/results \
  --dest-results cg_cascade/results_public
```

The sanitizer removes per-sample records, prompt excerpts, qualitative examples, judge reasons, and local home paths.

## Security Audit

Run the artifact leak check before committing or publishing:

```bash
python tests/test_no_sensitive_files.py
```

See `docs/security_review.md` for the release checklist and artifact policy.
