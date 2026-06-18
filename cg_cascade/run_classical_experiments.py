#!/usr/bin/env python
"""Classical detector baselines for the revised cost-aware triage prompt-injection study."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "datasets"
MODELS = ROOT / "models" / "saved"
RESULTS = ROOT / "results"

PRIMARY_TRAIN = DATA / "promptshield_train.parquet"
PRIMARY_VALIDATION = DATA / "promptshield_validation.parquet"
PRIMARY_TEST = DATA / "promptshield_test.parquet"
EXTERNAL_DATASETS = {
    "deepset_semantic_all": DATA / "deepset_semantic_all.parquet",
    "notinject_hard_negatives": DATA / "notinject_hard_negatives.parquet",
    "lakera_gandalf_attack_only": DATA / "lakera_gandalf_attack_only.parquet",
}


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, score: np.ndarray | None = None) -> dict[str, Any]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(tn / max(1, tn + fp)),
        "fpr": float(fp / max(1, tn + fp)),
        "fnr": float(fn / max(1, fn + tp)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "n_samples": int(len(y_true)),
        "positive_rate_pred": float(np.mean(y_pred)),
    }
    if score is not None:
        out["brier"] = float(brier_score_loss(y_true, score))
        out["ece_10"] = expected_calibration_error(y_true, score, n_bins=10)
        if len(np.unique(y_true)) == 2:
            out["roc_auc"] = float(roc_auc_score(y_true, score))
            out["average_precision"] = float(average_precision_score(y_true, score))
    return out


def expected_calibration_error(y_true: np.ndarray, score: np.ndarray, n_bins: int) -> float:
    y_true = np.asarray(y_true, dtype=int)
    score = np.asarray(score, dtype=float)
    total = len(score)
    if total == 0:
        return 0.0
    ece = 0.0
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    for left, right in zip(edges[:-1], edges[1:], strict=True):
        if math.isclose(right, 1.0):
            mask = (score >= left) & (score <= right)
        else:
            mask = (score >= left) & (score < right)
        if not np.any(mask):
            continue
        confidence = float(np.mean(score[mask]))
        accuracy = float(np.mean(y_true[mask]))
        ece += float(np.mean(mask)) * abs(confidence - accuracy)
    return float(ece)


def thresholds_from_validation(y_true: np.ndarray, score: np.ndarray) -> dict[str, Any]:
    candidates = np.unique(np.concatenate([np.linspace(0.0, 1.0, 1001), score]))
    rows = []
    for threshold in candidates:
        pred = (score >= threshold).astype(int)
        row = binary_metrics(y_true, pred, score)
        row["threshold"] = float(threshold)
        rows.append(row)

    best_f1 = max(rows, key=lambda row: (row["f1"], row["recall"], row["precision"]))

    def high_recall(target: float) -> dict[str, Any]:
        feasible = [row for row in rows if row["recall"] >= target]
        if feasible:
            return max(feasible, key=lambda row: (row["threshold"], row["precision"], row["f1"]))
        return max(rows, key=lambda row: (row["recall"], row["precision"], row["f1"]))

    def safe_bypass(max_false_bypass_rate: float) -> dict[str, Any]:
        positives = y_true == 1
        total_pos = int(np.sum(positives))
        out_rows = []
        for tau in candidates:
            bypass = score <= tau
            false_bypass = int(np.sum(positives & bypass))
            row = {
                "tau_safe": float(tau),
                "bypass_rate": float(np.mean(bypass)),
                "llm_call_rate": float(1.0 - np.mean(bypass)),
                "false_bypass_count": false_bypass,
                "false_bypass_rate_among_malicious": float(false_bypass / max(1, total_pos)),
            }
            out_rows.append(row)
        feasible = [
            row
            for row in out_rows
            if row["false_bypass_rate_among_malicious"] <= max_false_bypass_rate
        ]
        if feasible:
            return max(feasible, key=lambda row: (row["bypass_rate"], -row["false_bypass_count"]))
        return min(out_rows, key=lambda row: row["false_bypass_rate_among_malicious"])

    return {
        "classification": {
            "threshold_0_5": next(row for row in rows if math.isclose(row["threshold"], 0.5)),
            "best_f1": best_f1,
            "recall_ge_0_99": high_recall(0.99),
            "recall_ge_0_995": high_recall(0.995),
        },
        "triage": {
            "max_fb_rate_0": safe_bypass(0.0),
            "max_fb_rate_0_005": safe_bypass(0.005),
            "max_fb_rate_0_01": safe_bypass(0.01),
        },
    }


def evaluate_at_thresholds(y_true: np.ndarray, score: np.ndarray, thresholds: dict[str, Any]) -> dict[str, Any]:
    out = {
        "score_summary": {
            "min": float(np.min(score)),
            "p01": float(np.quantile(score, 0.01)),
            "p05": float(np.quantile(score, 0.05)),
            "p50": float(np.quantile(score, 0.50)),
            "p95": float(np.quantile(score, 0.95)),
            "p99": float(np.quantile(score, 0.99)),
            "max": float(np.max(score)),
        },
        "classification": {},
        "triage": {},
    }
    for name, row in thresholds["classification"].items():
        threshold = float(row["threshold"])
        out["classification"][name] = {
            "threshold": threshold,
            **binary_metrics(y_true, (score >= threshold).astype(int), score),
        }
    for name, row in thresholds["triage"].items():
        tau = float(row["tau_safe"])
        bypass = score <= tau
        positives = y_true == 1
        false_bypass = int(np.sum(positives & bypass))
        out["triage"][name] = {
            "tau_safe": tau,
            "bypass_rate": float(np.mean(bypass)),
            "llm_call_rate": float(1.0 - np.mean(bypass)),
            "bypass_count": int(np.sum(bypass)),
            "false_bypass_count": false_bypass,
            "false_bypass_rate_among_malicious": float(false_bypass / max(1, int(np.sum(positives)))),
        }
    return out


def bootstrap_ci(
    y_true: np.ndarray,
    score: np.ndarray,
    threshold: float,
    *,
    seed: int,
    n_bootstrap: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    values: dict[str, list[float]] = {key: [] for key in ["accuracy", "precision", "recall", "f1", "fpr"]}
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        row = binary_metrics(y_true[idx], (score[idx] >= threshold).astype(int), score[idx])
        for key in values:
            values[key].append(float(row[key]))
    return {
        key: {
            "mean": float(np.mean(vals)),
            "ci95_low": float(np.quantile(vals, 0.025)),
            "ci95_high": float(np.quantile(vals, 0.975)),
        }
        for key, vals in values.items()
    }


def build_features(
    word: TfidfVectorizer,
    char: TfidfVectorizer,
    prompts: list[str],
    *,
    fit: bool,
):
    if fit:
        return hstack([word.fit_transform(prompts), char.fit_transform(prompts)], format="csr")
    return hstack([word.transform(prompts), char.transform(prompts)], format="csr")


def train_model(train_df: pd.DataFrame, validation_df: pd.DataFrame, c_values: list[float]) -> dict[str, Any]:
    word = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        max_features=300_000,
        sublinear_tf=True,
        lowercase=True,
    )
    char = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_features=300_000,
        sublinear_tf=True,
        lowercase=True,
    )
    x_train = build_features(word, char, train_df["prompt"].astype(str).tolist(), fit=True)
    x_val = build_features(word, char, validation_df["prompt"].astype(str).tolist(), fit=False)
    y_train = train_df["label"].to_numpy().astype(int)
    y_val = validation_df["label"].to_numpy().astype(int)

    candidates = []
    for c_value in c_values:
        clf = LogisticRegression(
            C=c_value,
            class_weight="balanced",
            max_iter=2000,
            n_jobs=-1,
            random_state=42,
            solver="saga",
        )
        clf.fit(x_train, y_train)
        score = clf.predict_proba(x_val)[:, 1]
        threshold_payload = thresholds_from_validation(y_val, score)
        candidates.append(
            {
                "C": float(c_value),
                "clf": clf,
                "validation_score": score,
                "validation_thresholds": threshold_payload,
                "selection_metric": threshold_payload["classification"]["best_f1"]["f1"],
                "selection_recall": threshold_payload["classification"]["best_f1"]["recall"],
            },
        )
    best = max(candidates, key=lambda row: (row["selection_metric"], row["selection_recall"]))
    return {
        "word": word,
        "char": char,
        "clf": best["clf"],
        "selected_C": best["C"],
        "validation_score": best["validation_score"],
        "validation_thresholds": best["validation_thresholds"],
        "candidate_summary": [
            {
                "C": row["C"],
                "selection_metric_best_f1": float(row["selection_metric"]),
                "selection_recall_at_best_f1": float(row["selection_recall"]),
            }
            for row in candidates
        ],
    }


def score_model(bundle: dict[str, Any], df: pd.DataFrame) -> np.ndarray:
    x = build_features(bundle["word"], bundle["char"], df["prompt"].astype(str).tolist(), fit=False)
    return bundle["clf"].predict_proba(x)[:, 1]


def label_counts(df: pd.DataFrame) -> dict[str, int]:
    return {str(int(k)): int(v) for k, v in df["label"].value_counts().sort_index().items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260604)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--output", type=Path, default=RESULTS / "triage_classical_promptshield.json")
    parser.add_argument("--model-output", type=Path, default=MODELS / "triage_tfidf_logreg_promptshield.pkl")
    parser.add_argument("--c-values", default="0.5,1.0,2.0,4.0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    MODELS.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    start = time.time()
    train_df = pd.read_parquet(PRIMARY_TRAIN)
    validation_df = pd.read_parquet(PRIMARY_VALIDATION)
    test_df = pd.read_parquet(PRIMARY_TEST)
    c_values = [float(item) for item in args.c_values.split(",") if item.strip()]

    bundle = train_model(train_df, validation_df, c_values)
    y_val = validation_df["label"].to_numpy().astype(int)
    val_score = bundle["validation_score"]
    thresholds = bundle["validation_thresholds"]
    validation_eval = evaluate_at_thresholds(y_val, val_score, thresholds)

    test_score = score_model(bundle, test_df)
    y_test = test_df["label"].to_numpy().astype(int)
    test_eval = evaluate_at_thresholds(y_test, test_score, thresholds)
    test_eval["bootstrap_ci_best_f1_threshold"] = bootstrap_ci(
        y_test,
        test_score,
        float(thresholds["classification"]["best_f1"]["threshold"]),
        seed=args.seed,
        n_bootstrap=args.bootstrap,
    )
    test_eval["bootstrap_ci_recall_ge_0_99_threshold"] = bootstrap_ci(
        y_test,
        test_score,
        float(thresholds["classification"]["recall_ge_0_99"]["threshold"]),
        seed=args.seed + 1,
        n_bootstrap=args.bootstrap,
    )

    external_eval = {}
    for name, path in EXTERNAL_DATASETS.items():
        df = pd.read_parquet(path)
        y = df["label"].to_numpy().astype(int)
        score = score_model(bundle, df)
        external_eval[name] = {
            "label_counts": label_counts(df),
            **evaluate_at_thresholds(y, score, thresholds),
        }

    payload = {
        "experiment": "triage_classical_promptshield",
        "detector": "word_char_tfidf_logistic_regression",
        "training_policy": {
            "train": str(PRIMARY_TRAIN),
            "validation_for_hyperparameters_and_thresholds": str(PRIMARY_VALIDATION),
            "test": str(PRIMARY_TEST),
            "excluded": "imoxto and HackAPrompt outcome labels are not used",
        },
        "selected_C": bundle["selected_C"],
        "candidate_summary": bundle["candidate_summary"],
        "data": {
            "train": {"n": int(len(train_df)), "label_counts": label_counts(train_df)},
            "validation": {"n": int(len(validation_df)), "label_counts": label_counts(validation_df)},
            "test": {"n": int(len(test_df)), "label_counts": label_counts(test_df)},
        },
        "thresholds_selected_on_validation": thresholds,
        "validation": validation_eval,
        "test": test_eval,
        "external": external_eval,
        "elapsed_seconds": float(time.time() - start),
    }
    joblib.dump({"word": bundle["word"], "char": bundle["char"], "clf": bundle["clf"], "thresholds": thresholds}, args.model_output)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "selected_C": payload["selected_C"],
        "validation_best_f1": validation_eval["classification"]["best_f1"],
        "test_best_f1_threshold": test_eval["classification"]["best_f1"],
        "test_recall_ge_0_99_threshold": test_eval["classification"]["recall_ge_0_99"],
        "test_safe_bypass_0_005": test_eval["triage"]["max_fb_rate_0_005"],
        "external": {
            name: {
                "best_f1_threshold": row["classification"]["best_f1"],
                "safe_bypass_0_005": row["triage"]["max_fb_rate_0_005"],
            }
            for name, row in external_eval.items()
        },
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
