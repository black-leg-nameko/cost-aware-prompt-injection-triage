#!/usr/bin/env python
"""Create public aggregate result artifacts without raw prompts or per-sample outputs."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


RESULT_JSONS = [
    "triage_dataset_audit.json",
    "triage_classical_promptshield.json",
    "triage_embedding_classifiers.json",
    "triage_ayub_public_rf_zero_shot.json",
    "triage_openai_promptshield_test_balanced2000.json",
    "triage_openai_promptshield_test_natural3000.json",
    "triage_openai_deepset_semantic_all.json",
    "triage_openai_notinject_hard_negatives.json",
    "triage_openai_lakera_gandalf_attack_only.json",
    "triage_subset_classifier_comparison.json",
    "triage_np_calibrated_triage.json",
    "triage_shift_failsafe.json",
    "triage_mixed_low_rate_shift_probe.json",
    "triage_adaptive_padding_attack.json",
    "triage_adaptive_padding_e2e.json",
    "triage_sparse_mix_audit_e2e.json",
    "triage_quarantine_session_e2e.json",
    "triage_sparse_adaptive_defense_experiments.json",
    "triage_monitor_sensitivity.json",
    "triage_probabilistic_audit_simulation.json",
    "triage_llm_false_negative_analysis.json",
]

RESULT_MARKDOWN = [
    "triage_experiment_report.md",
    "triage_np_calibrated_triage.md",
    "triage_probabilistic_audit_simulation.md",
    "triage_sparse_mix_audit_e2e.md",
    "triage_quarantine_session_e2e.md",
    "triage_sparse_adaptive_defense_experiments.md",
]

DROP_KEYS = {
    "records",
    "top_examples",
    "examples",
    "prompt",
    "prompt_excerpt",
    "judge_reason",
    "sample_ids",
    "adaptive_sample_ids",
}
LOCAL_PATH_PATTERNS = [
    re.compile(r"/(?:home|Users)/[^/\s\"']+(?:/[^\s\"']*)?"),
    re.compile(r"<local-home>(?:/[^\s\"']*)?"),
]
IDENTIFYING_TERMS = [
    "black" + "leg",
    "black-" + "leg-" + "nameko",
    "yosi" + "zuka",
    "yoshi" + "zuka",
    "\u5409\u585a",
]


def sanitize_text(value: str) -> str:
    text = value
    for pattern in LOCAL_PATH_PATTERNS:
        text = pattern.sub("<local-path>", text)
    for term in IDENTIFYING_TERMS:
        text = text.replace(term, "<anonymous>")
    return text


def sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: sanitize(val) for key, val in value.items() if key not in DROP_KEYS}
    if isinstance(value, list):
        return [sanitize(item) for item in value]
    if isinstance(value, str):
        return sanitize_text(value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-results", type=Path, required=True)
    parser.add_argument("--dest-results", type=Path, required=True)
    args = parser.parse_args()

    args.dest_results.mkdir(parents=True, exist_ok=True)
    for name in RESULT_JSONS:
        source = args.source_results / name
        if not source.exists():
            raise FileNotFoundError(source)
        payload = sanitize(json.loads(source.read_text(encoding="utf-8")))
        (args.dest_results / name).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    for name in RESULT_MARKDOWN:
        source = args.source_results / name
        if not source.exists():
            raise FileNotFoundError(source)
        text = sanitize_text(source.read_text(encoding="utf-8"))
        (args.dest_results / name).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
