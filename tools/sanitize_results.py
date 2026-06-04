#!/usr/bin/env python
"""Create public aggregate result artifacts without raw prompts or per-sample outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


RESULT_JSONS = [
    "iwsec_dataset_audit.json",
    "iwsec_classical_promptshield.json",
    "iwsec_embedding_classifiers.json",
    "iwsec_ayub_public_rf_zero_shot.json",
    "iwsec_openai_promptshield_test_balanced2000.json",
    "iwsec_openai_promptshield_test_natural3000.json",
    "iwsec_openai_deepset_semantic_all.json",
    "iwsec_openai_notinject_hard_negatives.json",
    "iwsec_openai_lakera_gandalf_attack_only.json",
    "iwsec_subset_classifier_comparison.json",
    "iwsec_shift_failsafe.json",
    "iwsec_mixed_low_rate_shift_probe.json",
    "iwsec_adaptive_padding_attack.json",
    "iwsec_llm_false_negative_analysis.json",
]

RESULT_MARKDOWN = [
    "iwsec_reworked_experiment_report.md",
]

DROP_KEYS = {
    "records",
    "top_examples",
    "examples",
    "prompt",
    "prompt_excerpt",
    "judge_reason",
    "sample_ids",
}
LOCAL_HOME_MARKER = "/home/" + "blackleg"


def sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: sanitize(val) for key, val in value.items() if key not in DROP_KEYS}
    if isinstance(value, list):
        return [sanitize(item) for item in value]
    if isinstance(value, str):
        return value.replace(LOCAL_HOME_MARKER, "<local-home>")
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
        text = source.read_text(encoding="utf-8").replace(LOCAL_HOME_MARKER, "<local-home>")
        (args.dest_results / name).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
