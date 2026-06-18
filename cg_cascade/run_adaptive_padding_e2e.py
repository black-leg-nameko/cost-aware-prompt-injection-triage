#!/usr/bin/env python
"""End-to-end judge evaluation on router-bypassed adaptive padding attacks."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack

from run_adaptive_padding_attack import (
    BLOCKED_TERMS,
    DATA,
    MIX_RATES,
    MODELS,
    N_BENIGN_FEATURES,
    REPEATS,
    SEED,
    TAU,
    benign_feature_padding,
    mix_stream,
    pad_attack_df,
    score_prompts,
    with_scores,
)
from run_openai_judge import load_dotenv, openai_judge
from run_shift_failsafe_experiment import REFERENCE, load_records


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def collect_adaptive_prompts(max_per_rate: int) -> list[dict[str, Any]]:
    bundle = joblib.load(MODELS / "triage_tfidf_logreg_promptshield.pkl")
    padding = benign_feature_padding(bundle, n_features=N_BENIGN_FEATURES, repeats=REPEATS)

    gandalf_df, gandalf_records = load_records(
        RESULTS / "triage_openai_lakera_gandalf_attack_only.json",
        DATA / "lakera_gandalf_attack_only.parquet",
        slice(None),
    )
    padded_df = pad_attack_df(gandalf_df, padding)
    padded_scores = score_prompts(bundle, padded_df["prompt"].astype(str).tolist())
    padded_records = with_scores(gandalf_records, padded_scores)

    ps_df, ps_records = load_records(
        RESULTS / "triage_openai_promptshield_test_natural3000.json",
        DATA / "promptshield_test.parquet",
        slice(1000, None),
    )
    rng = np.random.default_rng(SEED)
    rows: list[dict[str, Any]] = []
    for rate in MIX_RATES:
        n_attack = int(round(len(ps_records) * rate))
        attack_idx = rng.permutation(len(padded_records))[:n_attack]
        take = min(max_per_rate, n_attack)
        for local_index, global_index in enumerate(attack_idx[:take]):
            record = padded_records[int(global_index)]
            prompt = str(padded_df.iloc[int(global_index)]["prompt"])
            score = float(record["router_score"])
            rows.append(
                {
                    "mixture_rate": float(rate),
                    "attack_index": int(local_index),
                    "sample_id": str(record["sample_id"]),
                    "label": 1,
                    "router_score": score,
                    "router_bypass": bool(score <= TAU),
                    "prompt": prompt,
                },
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-per-rate", type=int, default=20)
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--base-url", default="https://api.openai.com/v1")
    parser.add_argument("--max-prompt-chars", type=int, default=1600)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--output", type=Path, default=RESULTS / "triage_adaptive_padding_e2e.json")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    started = time.time()
    rows = collect_adaptive_prompts(args.max_per_rate)

    cache: dict[str, dict[str, Any]] = {}
    if args.output.exists():
        prior = json.loads(args.output.read_text(encoding="utf-8"))
        cache = {str(r["sample_id"]): r for r in prior.get("records", [])}

    judged: list[dict[str, Any]] = []
    for row in rows:
        sid = str(row["sample_id"])
        if sid in cache and "llm_prediction" in cache[sid]:
            judged.append({**cache[sid], **row})
            continue
        result = openai_judge(str(row["prompt"]), args)
        out = {**row, "llm_prediction": int(result["judgement"]), "reason": result.get("reason", "")}
        judged.append(out)
        cache[sid] = out

    bypassed = [r for r in judged if r["router_bypass"]]
    judge_fn = sum(int(r["llm_prediction"]) == 0 for r in bypassed)
    judge_tp = sum(int(r["llm_prediction"]) == 1 for r in bypassed)
    by_rate: dict[str, Any] = {}
    for rate in MIX_RATES:
        sub = [r for r in bypassed if int(round(r["mixture_rate"] * 100)) == int(round(rate * 100))]
        if not sub:
            continue
        fn = sum(int(r["llm_prediction"]) == 0 for r in sub)
        by_rate[f"{int(rate * 100)}pct"] = {
            "n": len(sub),
            "router_bypass_count": len(sub),
            "judge_would_miss": fn,
            "judge_would_catch": len(sub) - fn,
            "judge_miss_rate_if_escalated": float(fn / len(sub)),
            "observed_e2e_fn_rate": 1.0,
            "counterfactual_e2e_fn_rate": float(fn / len(sub)),
        }

    payload = {
        "experiment": "triage_adaptive_padding_e2e",
        "status": "completed",
        "model": args.model,
        "tau_safe": TAU,
        "seed": SEED,
        "n_evaluated": len(judged),
        "interpretation": (
            "Adaptive padded attacks bypass the router at tau_NP. observed_e2e_fn_rate "
            "counts router bypass as end-to-end failure. counterfactual_e2e_fn_rate "
            "is the judge false-negative rate if those bypassed prompts were escalated."
        ),
        "summary": {
            "router_bypass_count": len(bypassed),
            "judge_would_miss": judge_fn,
            "judge_would_catch": judge_tp,
            "judge_miss_rate_if_escalated": float(judge_fn / max(1, len(bypassed))),
            "observed_e2e_fn_rate": 1.0,
            "counterfactual_e2e_fn_rate": float(judge_fn / max(1, len(bypassed))),
        },
        "by_mixture_rate": by_rate,
        "records": judged,
        "elapsed_seconds": float(time.time() - started),
    }
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))
    print(json.dumps(payload["by_mixture_rate"], indent=2))


if __name__ == "__main__":
    main()
