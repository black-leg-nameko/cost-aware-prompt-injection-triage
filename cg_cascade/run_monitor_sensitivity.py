#!/usr/bin/env python
"""Sensitivity of shift-monitor thresholds on cost-aware triage streams, including sparse padded mixes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from run_adaptive_padding_attack import (
    DATA,
    MIX_RATES,
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
from run_shift_failsafe_experiment import (
    NP_CAL_TAU,
    REFERENCE,
    RESULTS,
    STREAMS,
    evaluate_policy,
    extract_features,
    load_records,
    population_stability_index,
)


ROOT = Path(__file__).resolve().parent


def detect_shift_with_thresholds(
    reference: dict[str, np.ndarray],
    current: dict[str, np.ndarray],
    *,
    psi_threshold: float,
    ks_pvalue_threshold: float,
    keyword_diff_threshold: float,
) -> dict[str, Any]:
    score_psi = population_stability_index(reference["router_score"], current["router_score"])
    ks = ks_2samp(reference["prompt_length"], current["prompt_length"])
    ref_keyword = float(np.mean(reference["keyword_hit"]))
    cur_keyword = float(np.mean(current["keyword_hit"]))
    keyword_diff = cur_keyword - ref_keyword
    triggers = {
        "router_score_psi": score_psi >= psi_threshold,
        "prompt_length_ks": float(ks.pvalue) < ks_pvalue_threshold,
        "keyword_rate_drift": abs(keyword_diff) >= keyword_diff_threshold,
    }
    return {
        "shift_detected": bool(any(triggers.values())),
        "triggers": triggers,
        "router_score_psi": float(score_psi),
        "prompt_length_ks_pvalue": float(ks.pvalue),
        "keyword_rate_diff": float(keyword_diff),
    }


def sparse_padded_streams() -> dict[str, tuple[pd.DataFrame, list[dict[str, Any]]]]:
    bundle = joblib.load(MODELS / "triage_tfidf_logreg_promptshield.pkl")
    padding = benign_feature_padding(bundle, n_features=N_BENIGN_FEATURES, repeats=REPEATS)
    gandalf_df, gandalf_records = load_records(
        RESULTS / "triage_openai_lakera_gandalf_attack_only.json",
        DATA / "lakera_gandalf_attack_only.parquet",
        slice(None),
    )
    padded_gandalf_df = pad_attack_df(gandalf_df, padding)
    padded_scores = score_prompts(bundle, padded_gandalf_df["prompt"].astype(str).tolist())
    padded_gandalf_records = with_scores(gandalf_records, padded_scores)
    ps_df, ps_records = load_records(
        RESULTS / "triage_openai_promptshield_test_natural3000.json",
        DATA / "promptshield_test.parquet",
        slice(1000, None),
    )
    rng = np.random.default_rng(SEED)
    out: dict[str, tuple[pd.DataFrame, list[dict[str, Any]]]] = {}
    for rate in MIX_RATES:
        mixed_df, mixed_records, _ = mix_stream(
            ps_df,
            ps_records,
            padded_gandalf_df,
            padded_gandalf_records,
            rate=rate,
            rng=rng,
        )
        name = f"ps_holdout_plus_padded_adaptive_{int(rate * 100)}pct"
        out[name] = (mixed_df, mixed_records)
    return out


def main() -> None:
    ref_df, ref_records = load_records(REFERENCE["result"], REFERENCE["dataset"], REFERENCE["slice"])
    ref_features = extract_features(ref_df, ref_records)

    psi_thresholds = [0.10, 0.20, 0.30, 0.40]
    ks_p = 0.01
    kw = 0.15

    rows: list[dict[str, Any]] = []
    stream_specs: dict[str, tuple[pd.DataFrame, list[dict[str, Any]], float]] = {}
    for stream_name, spec in STREAMS.items():
        df, records = load_records(spec["result"], spec["dataset"], spec["slice"])
        stream_specs[stream_name] = (df, records, float(spec["tau"]))
    for stream_name, (df, records) in sparse_padded_streams().items():
        stream_specs[stream_name] = (df, records, NP_CAL_TAU)

    for stream_name, (df, records, tau) in stream_specs.items():
        features = extract_features(df, records)
        for psi_threshold in psi_thresholds:
            monitor = detect_shift_with_thresholds(
                ref_features,
                features,
                psi_threshold=psi_threshold,
                ks_pvalue_threshold=ks_p,
                keyword_diff_threshold=kw,
            )
            naive = evaluate_policy(records, tau, fail_safe=False)
            safe = evaluate_policy(records, tau, fail_safe=monitor["shift_detected"])
            rows.append(
                {
                    "stream": stream_name,
                    "psi_threshold": psi_threshold,
                    "ks_pvalue_threshold": ks_p,
                    "keyword_diff_threshold": kw,
                    "shift_detected": monitor["shift_detected"],
                    "router_score_psi": monitor["router_score_psi"],
                    "keyword_rate_diff": monitor["keyword_rate_diff"],
                    "naive_call_reduction": naive["cost_reduction"],
                    "naive_false_bypass_rate": naive["false_bypass_rate_among_malicious"],
                    "final_call_reduction": safe["cost_reduction"],
                    "final_false_bypass_rate": safe["false_bypass_rate_among_malicious"],
                },
            )

    payload = {
        "experiment": "triage_monitor_sensitivity",
        "includes_sparse_padded_mixtures": True,
        "psi_thresholds": psi_thresholds,
        "rows": rows,
    }
    out = RESULTS / "triage_monitor_sensitivity.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(out)
    for stream_name in stream_specs:
        sub = [r for r in rows if r["stream"] == stream_name]
        print(stream_name, "max PSI", max(r["router_score_psi"] for r in sub), "any shift", any(r["shift_detected"] for r in sub))


if __name__ == "__main__":
    main()
