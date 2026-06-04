#!/usr/bin/env python
"""Probe sparse OOD attack mixtures against the stream-level shift monitor."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from run_iwsec_shift_failsafe_experiment import (
    DATA,
    RESULTS,
    REFERENCE,
    detect_shift,
    evaluate_policy,
    extract_features,
    label_counts,
    load_records,
)


RATES = [0.01, 0.05, 0.10]
SEED = 42
TAU = 0.03


def mix_stream(
    id_df: pd.DataFrame,
    id_records: list[dict[str, Any]],
    ood_df: pd.DataFrame,
    ood_records: list[dict[str, Any]],
    *,
    rate: float,
    id_perm: np.ndarray,
    ood_perm: np.ndarray,
) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, int]]:
    n_ood = int(round(len(id_records) * rate))
    n_id = len(id_records) - n_ood
    id_idx = id_perm[:n_id]
    ood_idx = ood_perm[:n_ood]
    mixed_df = pd.concat([id_df.iloc[id_idx], ood_df.iloc[ood_idx]], ignore_index=True)
    mixed_records = [id_records[int(i)] for i in id_idx] + [ood_records[int(i)] for i in ood_idx]
    return mixed_df, mixed_records, {"id": int(n_id), "ood": int(n_ood)}


def main() -> None:
    ref_df, ref_records = load_records(REFERENCE["result"], REFERENCE["dataset"], REFERENCE["slice"])
    ref_features = extract_features(ref_df, ref_records)

    ps_df, ps_records = load_records(
        RESULTS / "iwsec_openai_promptshield_test_natural3000.json",
        DATA / "promptshield_test.parquet",
        slice(1000, None),
    )
    gandalf_df, gandalf_records = load_records(
        RESULTS / "iwsec_openai_lakera_gandalf_attack_only.json",
        DATA / "lakera_gandalf_attack_only.parquet",
        slice(None),
    )

    rng = np.random.default_rng(SEED)
    ps_perm = rng.permutation(len(ps_records))
    gandalf_perm = rng.permutation(len(gandalf_records))

    streams: dict[str, Any] = {}
    for rate in RATES:
        mixed_df, mixed_records, counts = mix_stream(
            ps_df,
            ps_records,
            gandalf_df,
            gandalf_records,
            rate=rate,
            id_perm=ps_perm,
            ood_perm=gandalf_perm,
        )
        monitor = detect_shift(ref_features, extract_features(mixed_df, mixed_records))
        naive = evaluate_policy(mixed_records, TAU, fail_safe=False)
        fail_safe = evaluate_policy(mixed_records, TAU, fail_safe=monitor["shift_detected"])
        streams[f"promptshield_holdout_plus_gandalf_{int(rate * 100)}pct"] = {
            "mixture_rate": float(rate),
            "component_counts": counts,
            "label_counts": label_counts(mixed_records),
            "tau_safe": TAU,
            "monitor": monitor,
            "naive_triage": naive,
            "failsafe_triage": fail_safe,
        }

    payload = {
        "experiment": "iwsec_mixed_low_rate_shift_probe",
        "seed": SEED,
        "reference": "PromptShield natural first 1,000 prompts, unlabeled for monitoring",
        "id_stream": "PromptShield natural holdout",
        "ood_stream": "Lakera Gandalf attacks",
        "notes": (
            "Diagnostic probe for sparse OOD attacks mixed into otherwise in-distribution traffic; "
            "not used for threshold tuning or primary evaluation."
        ),
        "streams": streams,
    }
    out = RESULTS / "iwsec_mixed_low_rate_shift_probe.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
