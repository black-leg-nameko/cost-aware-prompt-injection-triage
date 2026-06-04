#!/usr/bin/env python
"""Evaluate local classifiers on the exact PromptShield subsets judged by the LLM."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "datasets"
RESULTS = ROOT / "results"
MODELS = ROOT / "models" / "saved"
CACHE = ROOT / "embeddings" / "cache"

PROMPTSHIELD_TEST = DATA / "promptshield_test.parquet"
PROMPTSHIELD_TEST_EMBED = CACHE / "iwsec_promptshield_test_text_embedding_3_small.npy"

SUBSETS = {
    "balanced_2000": RESULTS / "iwsec_openai_promptshield_test_balanced2000.json",
    "natural_3000": RESULTS / "iwsec_openai_promptshield_test_natural3000.json",
}


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(tn / max(1, tn + fp)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "n_samples": int(len(y_true)),
    }


def positive_scores(model: Any, x: np.ndarray) -> np.ndarray:
    class_index = list(model.classes_).index(1)
    return model.predict_proba(x)[:, class_index]


def tfidf_scores(bundle: dict[str, Any], prompts: list[str]) -> np.ndarray:
    x_word = bundle["word"].transform(prompts)
    x_char = bundle["char"].transform(prompts)
    x = hstack([x_word, x_char], format="csr")
    return positive_scores(bundle["clf"], x)


def embedding_scores(bundle: dict[str, Any], x: np.ndarray) -> np.ndarray:
    if "scaler" in bundle:
        x = bundle["scaler"].transform(x)
    return positive_scores(bundle["clf"], x)


def evaluate_detector(
    *,
    name: str,
    scores: np.ndarray,
    y_true: np.ndarray,
    thresholds: dict[str, Any],
) -> dict[str, Any]:
    rows = {}
    for threshold_name in ["best_f1", "recall_ge_0_99"]:
        threshold = float(thresholds["classification"][threshold_name]["threshold"])
        rows[threshold_name] = {
            "threshold": threshold,
            **metrics(y_true, (scores >= threshold).astype(int)),
        }
    return {"detector": name, "classification": rows}


def load_subset(path: Path, test_by_id: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    sample_ids = [str(row["sample_id"]) for row in payload["records"]]
    subset = test_by_id.loc[sample_ids].reset_index(drop=True)
    labels_from_records = np.asarray([int(row["label"]) for row in payload["records"]], dtype=int)
    labels_from_df = subset["label"].to_numpy().astype(int)
    if not np.array_equal(labels_from_records, labels_from_df):
        raise ValueError(f"Label mismatch while aligning {path}")
    return subset, labels_from_records, payload


def main() -> None:
    test_df = pd.read_parquet(PROMPTSHIELD_TEST).reset_index(drop=True)
    if "sample_id" not in test_df.columns:
        raise ValueError("PromptShield test data must contain sample_id")
    test_by_id = test_df.set_index("sample_id", drop=False)
    test_embeddings = np.load(PROMPTSHIELD_TEST_EMBED)
    if len(test_embeddings) != len(test_df):
        raise ValueError("PromptShield test embeddings are not aligned with test data")

    row_index_by_id = {str(row.sample_id): int(idx) for idx, row in test_df.iterrows()}
    tfidf = joblib.load(MODELS / "iwsec_tfidf_logreg_promptshield.pkl")
    emb_lr = joblib.load(MODELS / "iwsec_openai_embedding_logreg.pkl")
    emb_rf = joblib.load(MODELS / "iwsec_openai_embedding_random_forest.pkl")

    results: dict[str, Any] = {
        "experiment": "iwsec_subset_classifier_comparison",
        "purpose": "Compare local classifiers and LLM judge on identical PromptShield sampled subsets.",
        "alignment_key": "sample_id",
        "subsets": {},
    }

    for subset_name, subset_path in SUBSETS.items():
        subset, y_true, llm_payload = load_subset(subset_path, test_by_id)
        prompts = subset["prompt"].astype(str).tolist()
        emb_indices = np.asarray([row_index_by_id[str(sample_id)] for sample_id in subset["sample_id"]], dtype=int)
        x_embed = test_embeddings[emb_indices]

        tfidf_score = tfidf_scores(tfidf, prompts)
        emb_lr_score = embedding_scores(emb_lr, x_embed)
        emb_rf_score = embedding_scores(emb_rf, x_embed)
        llm_pred = np.asarray([int(row["llm_prediction"]) for row in llm_payload["records"]], dtype=int)

        results["subsets"][subset_name] = {
            "source": str(subset_path),
            "n_samples": int(len(subset)),
            "label_counts": {str(int(k)): int(v) for k, v in subset["label"].value_counts().sort_index().items()},
            "llm_judge": metrics(y_true, llm_pred),
            "detectors": [
                evaluate_detector(
                    name="TF-IDF LR",
                    scores=tfidf_score,
                    y_true=y_true,
                    thresholds=tfidf["thresholds"],
                ),
                evaluate_detector(
                    name="OpenAI embedding LR",
                    scores=emb_lr_score,
                    y_true=y_true,
                    thresholds=emb_lr["thresholds"],
                ),
                evaluate_detector(
                    name="OpenAI embedding RF",
                    scores=emb_rf_score,
                    y_true=y_true,
                    thresholds=emb_rf["thresholds"],
                ),
            ],
        }

    out = RESULTS / "iwsec_subset_classifier_comparison.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
