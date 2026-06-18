#!/usr/bin/env python
"""Source-matched NP/Clopper-Pearson calibration for IWSEC triage.

This script does not call external APIs. It reuses the saved TF-IDF router and
checkpointed LLM-judge outputs, then reserves PromptShield test examples that
are not part of the judged evaluation subsets as a labeled deployment
calibration block.
"""

from __future__ import annotations

import json
import math
import hashlib
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from scipy.stats import beta
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "datasets"
MODELS = ROOT / "models" / "saved"
RESULTS = ROOT / "results"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def pct(x: float) -> str:
    return f"{100.0 * x:.1f}%"


def clopper_upper(k: int, n: int, alpha: float) -> float:
    if n <= 0:
        return 1.0
    if k >= n:
        return 1.0
    return float(beta.ppf(1.0 - alpha, k + 1, n - k))


def wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    phat = k / n
    denom = 1.0 + z * z / n
    center = (phat + z * z / (2.0 * n)) / denom
    half = z * math.sqrt(phat * (1.0 - phat) / n + z * z / (4.0 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(tn / max(1, tn + fp)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def tfidf_scores(df: pd.DataFrame, model: dict[str, Any]) -> np.ndarray:
    text = df["prompt"].astype(str).tolist()
    x = hstack([model["word"].transform(text), model["char"].transform(text)], format="csr")
    return model["clf"].predict_proba(x)[:, 1]


def max_allowed_fb(n_pos: int, epsilon: float, alpha: float) -> int:
    allowed = -1
    for k in range(n_pos + 1):
        if clopper_upper(k, n_pos, alpha) <= epsilon:
            allowed = k
        else:
            # The one-sided upper bound is monotone in k, so we can stop.
            if allowed >= 0:
                break
    return allowed


def required_positives_for_allowed_k(k_allowed: int, epsilon: float, alpha: float, limit: int = 50_000) -> int | None:
    for n_pos in range(1, limit + 1):
        if clopper_upper(k_allowed, n_pos, alpha) <= epsilon:
            return n_pos
    return None


def choose_threshold(scores: np.ndarray, labels: np.ndarray, epsilon: float, alpha: float) -> dict[str, Any]:
    pos_scores = np.sort(scores[labels == 1])
    n_pos = int(len(pos_scores))
    k_allowed = max_allowed_fb(n_pos, epsilon, alpha)
    if k_allowed < 0:
        tau = 0.0
        k_cal = int(np.sum((scores <= tau) & (labels == 1)))
    elif k_allowed >= n_pos:
        tau = 1.0
        k_cal = n_pos
    elif k_allowed == 0:
        tau = float(np.nextafter(pos_scores[0], -np.inf))
        k_cal = 0
    else:
        tau = float(np.nextafter(pos_scores[k_allowed], -np.inf))
        k_cal = int(np.sum((scores <= tau) & (labels == 1)))
    return {
        "tau_safe": tau,
        "epsilon": float(epsilon),
        "alpha": float(alpha),
        "confidence": float(1.0 - alpha),
        "calibration_positives": n_pos,
        "max_allowed_false_bypass": int(k_allowed),
        "calibration_false_bypass_count": int(k_cal),
        "calibration_false_bypass_rate": float(k_cal / max(1, n_pos)),
        "calibration_cp_upper": clopper_upper(k_cal, n_pos, alpha),
        "calibration_bypass_count": int(np.sum(scores <= tau)),
        "calibration_call_reduction": float(np.mean(scores <= tau)),
    }


def fixed_threshold_positive_risk(scores: np.ndarray, labels: np.ndarray, tau: float, alpha: float) -> dict[str, Any]:
    n_pos = int(np.sum(labels == 1))
    k = int(np.sum((scores <= tau) & (labels == 1)))
    return {
        "n": int(len(scores)),
        "positives": n_pos,
        "tau_safe": float(tau),
        "false_bypass_count": k,
        "false_bypass_rate": float(k / max(1, n_pos)),
        "cp_upper": clopper_upper(k, n_pos, alpha),
        "call_reduction": float(np.mean(scores <= tau)),
    }


def hash_split_checks(calibration_df: pd.DataFrame, scores: np.ndarray, labels: np.ndarray, epsilon: float, alpha: float) -> list[dict[str, Any]]:
    split_bits = np.array(
        [
            int(hashlib.sha256(str(sample_id).encode("utf-8")).hexdigest(), 16) % 2
            for sample_id in calibration_df["sample_id"].astype(str)
        ],
        dtype=int,
    )
    rows: list[dict[str, Any]] = []
    for selection_bit in [0, 1]:
        select = split_bits == selection_bit
        confirm = ~select
        selected = choose_threshold(scores[select], labels[select], epsilon, alpha)
        confirmed = fixed_threshold_positive_risk(scores[confirm], labels[confirm], selected["tau_safe"], alpha)
        rows.append(
            {
                "selection_bit": int(selection_bit),
                "selection": selected,
                "confirmation": confirmed,
                "confirmation_passes": bool(confirmed["cp_upper"] <= epsilon),
            },
        )
    conservative_tau = min(row["selection"]["tau_safe"] for row in rows)
    rows.append(
        {
            "selection_bit": "min_of_hash_halves",
            "selection": {"tau_safe": float(conservative_tau)},
            "confirmation": fixed_threshold_positive_risk(scores, labels, conservative_tau, alpha),
            "confirmation_passes": bool(
                fixed_threshold_positive_risk(scores, labels, conservative_tau, alpha)["cp_upper"] <= epsilon
            ),
        },
    )
    return rows


def evaluate_records(records: list[dict[str, Any]], tau: float) -> dict[str, Any]:
    labels = np.array([int(r["label"]) for r in records])
    llm = np.array([int(r["llm_prediction"]) for r in records])
    scores = np.array([float(r["router_score"]) for r in records])
    bypass = scores <= tau
    final_pred = llm.copy()
    final_pred[bypass] = 0
    out = metrics(labels, final_pred)
    fb = int(np.sum((labels == 1) & bypass))
    n_pos = int(np.sum(labels == 1))
    lo, hi = wilson(fb, n_pos)
    out.update(
        {
            "n": int(len(records)),
            "positives": n_pos,
            "tau_safe": float(tau),
            "llm_call_rate": float(1.0 - np.mean(bypass)),
            "cost_reduction": float(np.mean(bypass)),
            "bypass_count": int(np.sum(bypass)),
            "false_bypass_count": fb,
            "false_bypass_rate_among_malicious": float(fb / max(1, n_pos)),
            "false_bypass_wilson95": [lo, hi],
            "false_bypass_cp_upper95": clopper_upper(fb, n_pos, 0.05),
        },
    )
    return out


def evaluate_full_router(df: pd.DataFrame, scores: np.ndarray, tau: float) -> dict[str, Any]:
    labels = df["label"].to_numpy()
    bypass = scores <= tau
    fb = int(np.sum((labels == 1) & bypass))
    n_pos = int(np.sum(labels == 1))
    lo, hi = wilson(fb, n_pos)
    return {
        "n": int(len(df)),
        "positives": n_pos,
        "tau_safe": float(tau),
        "call_reduction": float(np.mean(bypass)),
        "false_bypass_count": fb,
        "false_bypass_rate_among_malicious": float(fb / max(1, n_pos)),
        "false_bypass_wilson95": [lo, hi],
        "false_bypass_cp_upper95": clopper_upper(fb, n_pos, 0.05),
    }


def main() -> None:
    model = joblib.load(MODELS / "iwsec_tfidf_logreg_promptshield.pkl")
    test_df = pd.read_parquet(DATA / "promptshield_test.parquet")
    natural = load_json(RESULTS / "iwsec_openai_promptshield_test_natural3000.json")
    balanced = load_json(RESULTS / "iwsec_openai_promptshield_test_balanced2000.json")

    eval_ids = {str(r["sample_id"]) for r in natural["records"]} | {str(r["sample_id"]) for r in balanced["records"]}
    candidate_df = test_df[~test_df["sample_id"].astype(str).isin(eval_ids)].copy()
    calibration_df = candidate_df.sample(n=10_000, random_state=20260606).sort_values("sample_id").reset_index(drop=True)
    calibration_ids = set(calibration_df["sample_id"].astype(str))
    test_excluding_calibration_df = test_df[~test_df["sample_id"].astype(str).isin(calibration_ids)].copy()
    unused_holdout_df = candidate_df[~candidate_df["sample_id"].astype(str).isin(calibration_ids)].copy()
    calibration_scores = tfidf_scores(calibration_df, model)
    calibration_labels = calibration_df["label"].to_numpy()

    thresholds = {
        "np_cp_eps_0_005_alpha_0_05": choose_threshold(calibration_scores, calibration_labels, 0.005, 0.05),
        "np_cp_eps_0_01_alpha_0_05": choose_threshold(calibration_scores, calibration_labels, 0.01, 0.05),
    }

    full_scores = tfidf_scores(test_df, model)
    test_excluding_calibration_scores = tfidf_scores(test_excluding_calibration_df, model)
    unused_holdout_scores = tfidf_scores(unused_holdout_df, model)
    payload = {
        "experiment": "iwsec_np_calibrated_triage",
        "router": "PromptShield-trained word/char TF-IDF logistic regression",
        "calibration_source": "PromptShield official test rows excluding all LLM-judged natural/balanced evaluation records",
        "calibration_sampling": {"n": int(len(calibration_df)), "random_state": 20260606},
        "calibration_label_counts": {str(k): int(v) for k, v in calibration_df["label"].value_counts().sort_index().items()},
        "evaluation_exclusion": {
            "llm_judged_evaluation_records_excluded_from_calibration": int(len(eval_ids)),
            "calibration_records_excluded_from_holdout_router_only": int(len(calibration_ids)),
            "test_excluding_calibration_n": int(len(test_excluding_calibration_df)),
            "unused_non_llm_holdout_n": int(len(unused_holdout_df)),
        },
        "procedure": (
            "Choose the largest tau whose one-sided Clopper-Pearson/order-statistic "
            "upper tolerance bound on Pr[s(X)<=tau | Y=1] is <= epsilon at "
            "confidence 1-alpha."
        ),
        "thresholds": thresholds,
        "positive_budget_requirements": {
            "epsilon": 0.005,
            "alpha": 0.05,
            "minimum_positives_for_allowed_false_bypasses": {
                str(k): required_positives_for_allowed_k(k, 0.005, 0.05)
                for k in [0, 1, 2, 5, 6, 10, 20]
            },
        },
        "hash_split_sensitivity": hash_split_checks(calibration_df, calibration_scores, calibration_labels, 0.005, 0.05),
        "evaluation": {},
    }

    for name, threshold in thresholds.items():
        tau = float(threshold["tau_safe"])
        payload["evaluation"][name] = {
            "natural_3000": evaluate_records(natural["records"], tau),
            "balanced_2000": evaluate_records(balanced["records"], tau),
            "full_promptshield_test_router_only": evaluate_full_router(test_df, full_scores, tau),
            "promptshield_test_excluding_calibration_router_only": evaluate_full_router(
                test_excluding_calibration_df,
                test_excluding_calibration_scores,
                tau,
            ),
            "unused_non_llm_holdout_router_only": evaluate_full_router(unused_holdout_df, unused_holdout_scores, tau),
        }

    out_json = RESULTS / "iwsec_np_calibrated_triage.json"
    write_json(out_json, payload)

    lines = [
        "# NP/Clopper-Pearson Source-Matched Triage Calibration",
        "",
        f"Calibration source: {payload['calibration_source']}.",
        f"Calibration n={len(calibration_df)}, labels={payload['calibration_label_counts']}.",
        "The threshold is the positive-score order statistic whose one-sided upper tolerance bound is below epsilon.",
        "",
        "| Policy | eps | alpha | tau | cal FB | cal CP upper | cal call red. | natural call red. | natural FB | natural F1 | balanced call red. | balanced FB | balanced F1 | non-cal test FB | unused holdout FB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, threshold in thresholds.items():
        nat = payload["evaluation"][name]["natural_3000"]
        bal = payload["evaluation"][name]["balanced_2000"]
        noncal = payload["evaluation"][name]["promptshield_test_excluding_calibration_router_only"]
        unused = payload["evaluation"][name]["unused_non_llm_holdout_router_only"]
        lines.append(
            "| "
            + " | ".join(
                [
                    name,
                    f"{threshold['epsilon']:.3f}",
                    f"{threshold['alpha']:.2f}",
                    f"{threshold['tau_safe']:.5f}",
                    f"{threshold['calibration_false_bypass_count']}/{threshold['calibration_positives']}",
                    pct(threshold["calibration_cp_upper"]),
                    pct(threshold["calibration_call_reduction"]),
                    pct(nat["cost_reduction"]),
                    f"{nat['false_bypass_count']}/{nat['positives']} ({pct(nat['false_bypass_rate_among_malicious'])})",
                    f"{nat['f1']:.3f}",
                    pct(bal["cost_reduction"]),
                    f"{bal['false_bypass_count']}/{bal['positives']} ({pct(bal['false_bypass_rate_among_malicious'])})",
                    f"{bal['f1']:.3f}",
                    f"{noncal['false_bypass_count']}/{noncal['positives']} ({pct(noncal['false_bypass_rate_among_malicious'])})",
                    f"{unused['false_bypass_count']}/{unused['positives']} ({pct(unused['false_bypass_rate_among_malicious'])})",
                ],
            )
            + " |",
        )
    lines.extend(
        [
            "",
            "## Calibration positive budget",
            "",
            "| Allowed calibration FB | Minimum positives for eps=0.005, alpha=0.05 |",
            "|---:|---:|",
        ],
    )
    for k, n_pos in payload["positive_budget_requirements"]["minimum_positives_for_allowed_false_bypasses"].items():
        lines.append(f"| {k} | {n_pos} |")
    lines.extend(
        [
            "",
            "## Hash split sensitivity",
            "",
            "| Selection half | Tau | Select FB | Confirm FB | Confirm CP upper | Confirm pass |",
            "|---|---:|---:|---:|---:|---|",
        ],
    )
    for row in payload["hash_split_sensitivity"]:
        sel = row["selection"]
        conf = row["confirmation"]
        select_fb = "--"
        if "calibration_false_bypass_count" in sel:
            select_fb = f"{sel['calibration_false_bypass_count']}/{sel['calibration_positives']}"
        lines.append(
            f"| {row['selection_bit']} | {sel['tau_safe']:.5f} | {select_fb} | "
            f"{conf['false_bypass_count']}/{conf['positives']} | {pct(conf['cp_upper'])} | "
            f"{'yes' if row['confirmation_passes'] else 'no'} |",
        )
    out_md = RESULTS / "iwsec_np_calibrated_triage.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out_md.read_text(encoding="utf-8"))
    print(f"\nWrote {out_json}")


if __name__ == "__main__":
    main()
