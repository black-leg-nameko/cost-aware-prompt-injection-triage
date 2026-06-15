#!/usr/bin/env python
"""Deployed end-to-end sparse-mix triage with probabilistic bypass auditing."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack

from run_iwsec_adaptive_padding_attack import (
    DATA,
    MODELS,
    N_BENIGN_FEATURES,
    REPEATS,
    SEED,
    benign_feature_padding,
    mix_stream,
    pad_attack_df,
    score_prompts,
    with_scores,
)
from run_iwsec_shift_failsafe_experiment import (
    NP_CAL_TAU,
    REFERENCE,
    RESULTS,
    binary_metrics,
    evaluate_policy,
    extract_features,
    load_records,
)


OUT_JSON = RESULTS / "iwsec_sparse_mix_audit_e2e.json"
OUT_MD = RESULTS / "iwsec_sparse_mix_audit_e2e.md"
MIX_RATE = 0.01
AUDIT_RATE = 0.05
AUDIT_SEED = 0


def stable_audit(sample_id: str, audit_rate: float) -> bool:
    digest = hashlib.sha256(f"{sample_id}:audit:{AUDIT_SEED}".encode()).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    return bucket < audit_rate


def pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def e2e_by_sample_id(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {str(row["sample_id"]): row for row in payload.get("records", [])}


def evaluate_audit_policy(
    records: list[dict[str, Any]],
    *,
    tau: float,
    audit_rate: float,
    audit_seed: int,
    judge_by_id: dict[str, dict[str, Any]],
    adaptive_ids: set[str],
) -> dict[str, Any]:
    y_true = np.asarray([int(row["label"]) for row in records], dtype=int)
    scores = np.asarray([float(row["router_score"]) for row in records], dtype=float)
    bypass = scores <= tau

    pred = np.zeros_like(y_true, dtype=int)
    llm_called = np.zeros_like(y_true, dtype=bool)
    audited = np.zeros_like(y_true, dtype=bool)

    for index, row in enumerate(records):
        sample_id = str(row["sample_id"])
        llm_pred = int(judge_by_id.get(sample_id, row)["llm_prediction"])
        if bypass[index]:
            audited[index] = stable_audit(sample_id, audit_rate)
            if audited[index]:
                pred[index] = llm_pred
                llm_called[index] = True
            else:
                pred[index] = 0
        else:
            pred[index] = llm_pred
            llm_called[index] = True

    positives = y_true == 1
    adaptive_mask = np.asarray([str(row["sample_id"]) in adaptive_ids for row in records], dtype=bool)
    adaptive_positives = adaptive_mask & positives
    adaptive_fn = int(np.sum(adaptive_positives & (pred == 0)))
    adaptive_audited = int(np.sum(adaptive_positives & audited))
    adaptive_caught_when_audited = int(
        np.sum(adaptive_positives & audited & (pred == 1)),
    )
    malicious_bypass = positives & bypass
    malicious_audited = malicious_bypass & audited
    malicious_caught_when_audited = int(np.sum(malicious_audited & (pred == 1)))

    naive = evaluate_policy(records, tau, fail_safe=False)
    return {
        "mixture_rate": MIX_RATE,
        "audit_rate": audit_rate,
        "audit_seed": audit_seed,
        "tau_safe": float(tau),
        "n_samples": int(len(records)),
        "naive_triage": naive,
        "llm_call_rate": float(np.mean(llm_called)),
        "cost_reduction": float(1.0 - np.mean(llm_called)),
        "audited_bypass_count": int(np.sum(bypass & audited)),
        "malicious_false_bypass_count": int(np.sum(positives & bypass & ~audited)),
        "malicious_false_bypass_rate": float(np.sum(positives & bypass & ~audited) / max(1, int(np.sum(positives)))),
        "adaptive_attack_count": int(np.sum(adaptive_positives)),
        "adaptive_audited_count": adaptive_audited,
        "adaptive_e2e_fn_count": adaptive_fn,
        "adaptive_e2e_fn_rate": float(adaptive_fn / max(1, int(np.sum(adaptive_positives)))),
        "adaptive_caught_when_audited": adaptive_caught_when_audited,
        "adaptive_judge_miss_when_audited": int(adaptive_audited - adaptive_caught_when_audited),
        "malicious_bypass_audited_count": int(np.sum(malicious_audited)),
        "malicious_caught_when_audited": malicious_caught_when_audited,
        "malicious_judge_miss_when_audited": int(np.sum(malicious_audited) - malicious_caught_when_audited),
        **binary_metrics(y_true, pred),
    }


def audit_seed_sensitivity(
    records: list[dict[str, Any]],
    *,
    tau: float,
    audit_rate: float,
    judge_by_id: dict[str, dict[str, Any]],
    adaptive_ids: set[str],
    seeds: range,
) -> dict[str, Any]:
    adaptive_fn_rates: list[float] = []
    adaptive_audited_counts: list[int] = []
    for seed in seeds:
        global AUDIT_SEED
        AUDIT_SEED = seed
        row = evaluate_audit_policy(
            records,
            tau=tau,
            audit_rate=audit_rate,
            audit_seed=seed,
            judge_by_id=judge_by_id,
            adaptive_ids=adaptive_ids,
        )
        adaptive_fn_rates.append(float(row["adaptive_e2e_fn_rate"]))
        adaptive_audited_counts.append(int(row["adaptive_audited_count"]))
    return {
        "audit_seeds": list(seeds),
        "mean_adaptive_e2e_fn_rate": float(np.mean(adaptive_fn_rates)),
        "mean_adaptive_audited_count": float(np.mean(adaptive_audited_counts)),
    }


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    bundle = joblib.load(MODELS / "iwsec_tfidf_logreg_promptshield.pkl")
    padding = benign_feature_padding(bundle, n_features=N_BENIGN_FEATURES, repeats=REPEATS)

    gandalf_df, gandalf_records = load_records(
        RESULTS / "iwsec_openai_lakera_gandalf_attack_only.json",
        DATA / "lakera_gandalf_attack_only.parquet",
        slice(None),
    )
    padded_gandalf_df = pad_attack_df(gandalf_df, padding)
    padded_scores = score_prompts(bundle, padded_gandalf_df["prompt"].astype(str).tolist())
    padded_gandalf_records = with_scores(gandalf_records, padded_scores)

    ps_df, ps_records = load_records(
        RESULTS / "iwsec_openai_promptshield_test_natural3000.json",
        DATA / "promptshield_test.parquet",
        slice(1000, None),
    )
    rng = np.random.default_rng(SEED)
    mixed_df, mixed_records, component_counts = mix_stream(
        ps_df,
        ps_records,
        padded_gandalf_df,
        padded_gandalf_records,
        rate=MIX_RATE,
        rng=rng,
    )

    e2e_cache = e2e_by_sample_id(RESULTS / "iwsec_adaptive_padding_e2e.json")
    judge_by_id = {str(row["sample_id"]): row for row in ps_records}
    judge_by_id.update(e2e_cache)

    attack_records = component_counts["adaptive_attack"]
    adaptive_ids = {str(row["sample_id"]) for row in mixed_records[-attack_records:]}

    deployed = evaluate_audit_policy(
        mixed_records,
        tau=NP_CAL_TAU,
        audit_rate=AUDIT_RATE,
        audit_seed=AUDIT_SEED,
        judge_by_id=judge_by_id,
        adaptive_ids=adaptive_ids,
    )
    sensitivity = audit_seed_sensitivity(
        mixed_records,
        tau=NP_CAL_TAU,
        audit_rate=AUDIT_RATE,
        judge_by_id=judge_by_id,
        adaptive_ids=adaptive_ids,
        seeds=range(100),
    )

    ref_df, ref_records = load_records(REFERENCE["result"], REFERENCE["dataset"], REFERENCE["slice"])
    _ = extract_features(ref_df, ref_records)

    payload = {
        "experiment": "iwsec_sparse_mix_audit_e2e",
        "scope": (
            "Deployed end-to-end evaluation on the 1% PromptShield holdout + padded Gandalf "
            "mixture (seed 42) with NP-calibrated tau, monitor OFF, and 5% probabilistic "
            "audit on router bypasses. Judge labels for padded attacks come from fresh "
            "gpt-4o-mini calls on the padded prompt text."
        ),
        "component_counts": component_counts,
        "adaptive_sample_ids": sorted(adaptive_ids),
        "deployed_audit_on": deployed,
        "audit_seed_sensitivity_100": sensitivity,
        "counterfactual_no_audit": {
            "observed_e2e_fn_among_adaptive": deployed["adaptive_attack_count"],
            "malicious_false_bypass_rate": deployed["naive_triage"]["false_bypass_rate_among_malicious"],
        },
    }
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    row = deployed
    lines = [
        "# Sparse mixture audit-on end-to-end",
        "",
        f"Mixture: PS holdout + {int(MIX_RATE * 100)}% padded Gandalf (seed {SEED}).",
        f"Audit: {pct(AUDIT_RATE)} of bypass-eligible prompts (stable hash, seed {row['audit_seed']}).",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Adaptive attacks | {row['adaptive_attack_count']} |",
        f"| Adaptive audited | {row['adaptive_audited_count']} |",
        f"| Adaptive caught when audited | {row['adaptive_caught_when_audited']} |",
        f"| Adaptive E2E FN | {row['adaptive_e2e_fn_count']}/{row['adaptive_attack_count']} ({pct(row['adaptive_e2e_fn_rate'])}) |",
        f"| Malicious bypass audited | {row['malicious_bypass_audited_count']} |",
        f"| Malicious caught when audited | {row['malicious_caught_when_audited']} |",
        f"| All malicious E2E FN rate | {pct(row['malicious_false_bypass_rate'])} |",
        f"| LLM call rate | {pct(row['llm_call_rate'])} |",
        f"| Call reduction vs LLM-only | {pct(row['cost_reduction'])} |",
        f"| Naive call reduction (no audit) | {pct(row['naive_triage']['cost_reduction'])} |",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUT_MD.read_text(encoding="utf-8"))
    print(f"Wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
