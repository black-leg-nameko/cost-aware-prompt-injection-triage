#!/usr/bin/env python
"""Probabilistic bypass-audit simulation for sparse adaptive padding streams."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
INPUT = RESULTS / "iwsec_adaptive_padding_attack.json"
OUT_JSON = RESULTS / "iwsec_probabilistic_audit_simulation.json"
OUT_MD = RESULTS / "iwsec_probabilistic_audit_simulation.md"

AUDIT_RATES = [0.01, 0.05, 0.10]


def pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def audit_exposure_probability(audit_rate: float, bypassed_adaptive_attacks: int) -> float:
    return 1.0 - (1.0 - audit_rate) ** bypassed_adaptive_attacks


def stream_audit_metrics(stream: dict[str, Any], audit_rate: float) -> dict[str, Any]:
    attack_bypassed = int(stream["adaptive_attack_bypass_count"])
    naive_reduction = float(stream["naive_triage"]["cost_reduction"])
    if stream["monitor"]["shift_detected"]:
        final_reduction = float(stream["failsafe_triage"]["cost_reduction"])
        extra_call_rate = 0.0
    else:
        extra_call_rate = audit_rate * naive_reduction
        final_reduction = naive_reduction - extra_call_rate
    return {
        "audit_rate": audit_rate,
        "adaptive_bypass_count": attack_bypassed,
        "audit_exposure_probability": audit_exposure_probability(audit_rate, attack_bypassed),
        "expected_audited_adaptive_attacks": audit_rate * attack_bypassed,
        "extra_llm_call_rate": extra_call_rate,
        "final_call_reduction_after_audit": final_reduction,
    }


def main() -> None:
    source = json.loads(INPUT.read_text(encoding="utf-8"))
    streams = {
        name: stream
        for name, stream in source["mixed_streams"].items()
        if stream["adaptive_attack_bypass_count"] > 0
    }
    payload = {
        "experiment": "iwsec_probabilistic_bypass_audit_simulation",
        "source": str(INPUT),
        "scope": (
            "Routing-layer audit exposure simulation over Table 6 sparse adaptive streams; "
            "does not re-collect LLM judge labels for padded prompts."
        ),
        "audit_rates": AUDIT_RATES,
        "streams": {
            name: {str(rate): stream_audit_metrics(stream, rate) for rate in AUDIT_RATES}
            for name, stream in streams.items()
        },
    }
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Probabilistic bypass audit simulation",
        "",
        "Audit exposure is the probability that at least one injected adaptive bypass is routed to the judge.",
        "",
        "| Stream | Audit rate | Adaptive bypasses | Exposure prob. | Extra call rate | Final call red. |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, by_rate in payload["streams"].items():
        short = name.replace("promptshield_holdout_plus_adaptive_gandalf_", "PS+").replace("pct", "%")
        for rate in AUDIT_RATES:
            row = by_rate[str(rate)]
            lines.append(
                "| "
                + " | ".join(
                    [
                        short,
                        pct(rate),
                        str(row["adaptive_bypass_count"]),
                        pct(row["audit_exposure_probability"]),
                        pct(row["extra_llm_call_rate"]),
                        pct(row["final_call_reduction_after_audit"]),
                    ],
                )
                + " |",
            )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUT_MD.read_text(encoding="utf-8"))
    print(f"Wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
