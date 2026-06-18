#!/usr/bin/env python
"""Evaluate the public Ayub OpenAI-embedding RF on revised cost-aware triage datasets."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from run_embedding_experiments import EMBED, load_xy, positive_scores
from run_classical_experiments import (
    EXTERNAL_DATASETS,
    MODELS,
    PRIMARY_TEST,
    PRIMARY_VALIDATION,
    RESULTS,
    binary_metrics,
    evaluate_at_thresholds,
    label_counts,
    thresholds_from_validation,
)


def main() -> None:
    started = time.time()
    model = joblib.load(MODELS / "ayub_rf_openai.pkl")
    validation_df, x_val, y_val = load_xy(PRIMARY_VALIDATION, EMBED["promptshield_validation"])
    test_df, x_test, y_test = load_xy(PRIMARY_TEST, EMBED["promptshield_test"])
    val_score = positive_scores(model, x_val)
    thresholds = thresholds_from_validation(y_val, val_score)
    test_score = positive_scores(model, x_test)
    external: dict[str, Any] = {}
    for name, path in EXTERNAL_DATASETS.items():
        df, x, y = load_xy(path, EMBED[name])
        score = positive_scores(model, x)
        external[name] = {
            "label_counts": label_counts(df),
            **evaluate_at_thresholds(y, score, thresholds),
        }
    payload = {
        "experiment": "triage_ayub_public_rf_zero_shot",
        "detector": "ayub_public_rf_openai_embedding",
        "training_policy": "public model is not retrained; validation is used only for threshold selection",
        "validation": evaluate_at_thresholds(y_val, val_score, thresholds),
        "thresholds_selected_on_validation": thresholds,
        "test": evaluate_at_thresholds(y_test, test_score, thresholds),
        "external": external,
        "elapsed_seconds": float(time.time() - started),
    }
    out = RESULTS / "triage_ayub_public_rf_zero_shot.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "validation_best_f1": payload["validation"]["classification"]["best_f1"],
                "test_best_f1": payload["test"]["classification"]["best_f1"],
                "test_recall_ge_0_99": payload["test"]["classification"]["recall_ge_0_99"],
                "external_deepset_best_f1": payload["external"]["deepset_semantic_all"]["classification"]["best_f1"],
            },
            ensure_ascii=False,
            indent=2,
        ),
    )


if __name__ == "__main__":
    main()
