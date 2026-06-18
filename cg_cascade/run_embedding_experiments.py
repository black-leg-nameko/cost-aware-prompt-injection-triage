#!/usr/bin/env python
"""OpenAI-embedding classifier baselines for the revised cost-aware triage study."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from run_classical_experiments import (
    DATA,
    MODELS,
    RESULTS,
    EXTERNAL_DATASETS,
    PRIMARY_TEST,
    PRIMARY_TRAIN,
    PRIMARY_VALIDATION,
    binary_metrics,
    bootstrap_ci,
    evaluate_at_thresholds,
    label_counts,
    thresholds_from_validation,
)


CACHE = Path(__file__).resolve().parent / "embeddings" / "cache"
EMBED = {
    "promptshield_train": CACHE / "triage_promptshield_train_text_embedding_3_small.npy",
    "promptshield_validation": CACHE / "triage_promptshield_validation_text_embedding_3_small.npy",
    "promptshield_test": CACHE / "triage_promptshield_test_text_embedding_3_small.npy",
    "deepset_semantic_all": CACHE / "triage_deepset_semantic_all_text_embedding_3_small.npy",
    "notinject_hard_negatives": CACHE / "triage_notinject_hard_negatives_text_embedding_3_small.npy",
    "lakera_gandalf_attack_only": CACHE / "triage_lakera_gandalf_attack_only_text_embedding_3_small.npy",
}


def load_xy(df_path: Path, emb_path: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    df = pd.read_parquet(df_path).reset_index(drop=True)
    x = np.load(emb_path)
    if len(df) != len(x):
        raise ValueError(f"{df_path}: dataframe length {len(df)} != embedding length {len(x)}")
    y = df["label"].to_numpy().astype(int)
    return df, x, y


def positive_scores(model: Any, x: np.ndarray) -> np.ndarray:
    class_index = list(model.classes_).index(1)
    return model.predict_proba(x)[:, class_index]


def train_embedding_logreg(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    c_values: list[float],
) -> dict[str, Any]:
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_val_scaled = scaler.transform(x_val)
    candidates = []
    for c_value in c_values:
        clf = LogisticRegression(
            C=c_value,
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
            solver="lbfgs",
        )
        clf.fit(x_train_scaled, y_train)
        score = positive_scores(clf, x_val_scaled)
        thresholds = thresholds_from_validation(y_val, score)
        candidates.append(
            {
                "C": float(c_value),
                "clf": clf,
                "thresholds": thresholds,
                "score": score,
                "selection_metric": thresholds["classification"]["best_f1"]["f1"],
            },
        )
    best = max(candidates, key=lambda row: row["selection_metric"])
    return {
        "name": "openai_embedding_logreg",
        "scaler": scaler,
        "clf": best["clf"],
        "selected_C": best["C"],
        "validation_score": best["score"],
        "thresholds": best["thresholds"],
        "candidate_summary": [
            {"C": row["C"], "validation_best_f1": float(row["selection_metric"])}
            for row in candidates
        ],
    }


def train_embedding_rf(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    n_estimators: int,
) -> dict[str, Any]:
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample",
        max_features="sqrt",
    )
    clf.fit(x_train, y_train)
    score = positive_scores(clf, x_val)
    thresholds = thresholds_from_validation(y_val, score)
    return {
        "name": "openai_embedding_random_forest",
        "clf": clf,
        "n_estimators": int(n_estimators),
        "validation_score": score,
        "thresholds": thresholds,
    }


def score_bundle(bundle: dict[str, Any], x: np.ndarray) -> np.ndarray:
    if "scaler" in bundle:
        x = bundle["scaler"].transform(x)
    return positive_scores(bundle["clf"], x)


def evaluate_bundle(
    bundle: dict[str, Any],
    *,
    validation_df: pd.DataFrame,
    y_val: np.ndarray,
    test_df: pd.DataFrame,
    x_test: np.ndarray,
    y_test: np.ndarray,
    seed: int,
    bootstrap: int,
) -> dict[str, Any]:
    thresholds = bundle["thresholds"]
    validation = evaluate_at_thresholds(y_val, bundle["validation_score"], thresholds)
    test_score = score_bundle(bundle, x_test)
    test = evaluate_at_thresholds(y_test, test_score, thresholds)
    test["bootstrap_ci_best_f1_threshold"] = bootstrap_ci(
        y_test,
        test_score,
        float(thresholds["classification"]["best_f1"]["threshold"]),
        seed=seed,
        n_bootstrap=bootstrap,
    )
    external = {}
    for name, path in EXTERNAL_DATASETS.items():
        df, x, y = load_xy(path, EMBED[name])
        score = score_bundle(bundle, x)
        external[name] = {
            "label_counts": label_counts(df),
            **evaluate_at_thresholds(y, score, thresholds),
        }
    return {
        "detector": bundle["name"],
        "hyperparameters": {
            key: value
            for key, value in bundle.items()
            if key in {"selected_C", "n_estimators", "candidate_summary"}
        },
        "thresholds_selected_on_validation": thresholds,
        "validation": validation,
        "test": test,
        "external": external,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260604)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--c-values", default="0.1,0.5,1.0,2.0,4.0")
    parser.add_argument("--rf-estimators", type=int, default=300)
    parser.add_argument("--skip-rf", action="store_true")
    parser.add_argument("--output", type=Path, default=RESULTS / "triage_embedding_classifiers.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    MODELS.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    started = time.time()
    train_df, x_train, y_train = load_xy(PRIMARY_TRAIN, EMBED["promptshield_train"])
    validation_df, x_val, y_val = load_xy(PRIMARY_VALIDATION, EMBED["promptshield_validation"])
    test_df, x_test, y_test = load_xy(PRIMARY_TEST, EMBED["promptshield_test"])
    c_values = [float(item) for item in args.c_values.split(",") if item.strip()]

    bundles = [train_embedding_logreg(x_train, y_train, x_val, y_val, c_values)]
    if not args.skip_rf:
        bundles.append(train_embedding_rf(x_train, y_train, x_val, y_val, args.rf_estimators))

    results = []
    for bundle in bundles:
        result = evaluate_bundle(
            bundle,
            validation_df=validation_df,
            y_val=y_val,
            test_df=test_df,
            x_test=x_test,
            y_test=y_test,
            seed=args.seed,
            bootstrap=args.bootstrap,
        )
        results.append(result)
        model_path = MODELS / f"triage_{bundle['name']}.pkl"
        joblib.dump(bundle, model_path)
        result["model_path"] = str(model_path)

    payload = {
        "experiment": "triage_embedding_classifiers",
        "embedding_model": "text-embedding-3-small",
        "training_policy": {
            "train": str(PRIMARY_TRAIN),
            "validation_for_hyperparameters_and_thresholds": str(PRIMARY_VALIDATION),
            "test": str(PRIMARY_TEST),
        },
        "data": {
            "train": {"n": int(len(train_df)), "label_counts": label_counts(train_df)},
            "validation": {"n": int(len(validation_df)), "label_counts": label_counts(validation_df)},
            "test": {"n": int(len(test_df)), "label_counts": label_counts(test_df)},
        },
        "detectors": results,
        "elapsed_seconds": float(time.time() - started),
    }
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        result["detector"]: {
            "validation_best_f1": result["validation"]["classification"]["best_f1"],
            "test_best_f1": result["test"]["classification"]["best_f1"],
            "test_recall_ge_0_99": result["test"]["classification"]["recall_ge_0_99"],
            "test_safe_bypass_0": result["test"]["triage"]["max_fb_rate_0"],
            "external_deepset_best_f1": result["external"]["deepset_semantic_all"]["classification"]["best_f1"],
        }
        for result in results
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
