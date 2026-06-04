#!/usr/bin/env python
"""Summarize revised IWSEC dataset and experiment results into a paper-ready report."""

from __future__ import annotations

import json
import argparse
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results_public" if (ROOT / "results_public").exists() else ROOT / "results"


def load_json(name: str) -> dict[str, Any]:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS,
        help="Directory containing aggregate result JSONs. Defaults to results_public if present, otherwise results.",
    )
    return parser.parse_args()


def fmt(value: float) -> str:
    return f"{value:.3f}"


def pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def maybe_fmt(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "--"
    return f"{value:.{digits}f}"


def ks_text(value: float | None) -> str:
    if value is None:
        return "--"
    if value < 1e-200:
        return "<1e-200"
    if value < 1e-100:
        return "<1e-100"
    return f"{value:.3f}"


def metric_row(name: str, row: dict[str, Any]) -> str:
    cm = row["confusion_matrix"]
    return (
        f"| {name} | {fmt(row['accuracy'])} | {fmt(row['precision'])} | {fmt(row['recall'])} | "
        f"{fmt(row.get('specificity', 0.0))} | {fmt(row['f1'])} | "
        f"{cm['tn']}/{cm['fp']}/{cm['fn']}/{cm['tp']} |"
    )


def triage_row(name: str, row: dict[str, Any]) -> str:
    false_bypass = (
        "--"
        if name.startswith("NotInject")
        else f"{row['false_bypass_count']} ({pct(row['false_bypass_rate_among_malicious'])})"
    )
    return (
        f"| {name} | {row['policy']} | {pct(row['cost_reduction'])} | {pct(row['llm_call_rate'])} | "
        f"{fmt(row['accuracy'])} | {fmt(row['recall'])} | {fmt(row.get('specificity', 0.0))} | "
        f"{fmt(row['f1'])} | {false_bypass} |"
    )


def find_triage(payload: dict[str, Any], policy: str) -> dict[str, Any]:
    return next(row for row in payload["triage"] if row["policy"] == policy)


def detector_metric(result: dict[str, Any], detector_name: str, threshold_name: str = "best_f1") -> dict[str, Any]:
    for detector in result["detectors"]:
        if detector["detector"] == detector_name:
            return detector["test"]["classification"][threshold_name]
    raise KeyError(detector_name)


def detector_external(result: dict[str, Any], detector_name: str, dataset: str, threshold_name: str = "best_f1") -> dict[str, Any]:
    for detector in result["detectors"]:
        if detector["detector"] == detector_name:
            return detector["external"][dataset]["classification"][threshold_name]
    raise KeyError(detector_name)


def main() -> None:
    global RESULTS
    args = parse_args()
    RESULTS = args.results_dir

    audit = load_json("iwsec_dataset_audit.json")
    tfidf = load_json("iwsec_classical_promptshield.json")
    embedding = load_json("iwsec_embedding_classifiers.json")
    ayub = load_json("iwsec_ayub_public_rf_zero_shot.json")
    llm_bal = load_json("iwsec_openai_promptshield_test_balanced2000.json")
    llm_nat = load_json("iwsec_openai_promptshield_test_natural3000.json")
    llm_deepset = load_json("iwsec_openai_deepset_semantic_all.json")
    llm_notinject = load_json("iwsec_openai_notinject_hard_negatives.json")
    llm_gandalf = load_json("iwsec_openai_lakera_gandalf_attack_only.json")
    subset = load_json("iwsec_subset_classifier_comparison.json")
    shift = load_json("iwsec_shift_failsafe.json")
    mixed = load_json("iwsec_mixed_low_rate_shift_probe.json")
    fn_audit = load_json("iwsec_llm_false_negative_analysis.json")

    lines: list[str] = []
    lines.append("# Revised IWSEC Experiment Report")
    lines.append("")
    lines.append("## Label Policy")
    policy = audit["label_policy"]
    lines.append(f"- Primary task: {policy['primary_task']}.")
    lines.append(f"- Positive label: {policy['positive_label']}.")
    lines.append(f"- Negative label: {policy['negative_label']}.")
    lines.append(f"- Excluded: {policy['excluded']}.")
    lines.append("")
    lines.append("## Dataset Audit")
    lines.append("| Dataset | N | Label counts | Keyword hit | Duplicate prompts | Role |")
    lines.append("|---|---:|---:|---:|---:|---|")
    for key, row in audit["datasets"].items():
        note = row["label_note"]
        lines.append(
            f"| {key} | {row['n_samples']} | {row['label_counts']} | {pct(row['keyword_hit_rate'])} | "
            f"{pct(row['duplicate_prompt_rate'])} | {note['role']} |"
        )

    lines.append("")
    lines.append("## PromptShield Test: Detector Comparison")
    lines.append("Full-test classifier rows use all 23,516 PromptShield test prompts. LLM rows use sampled PromptShield test subsets.")
    lines.append("| Detector / setting | Acc | Prec | Rec | Spec | F1 | TN/FP/FN/TP |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    lines.append(metric_row("TF-IDF LR, validation best-F1 threshold", tfidf["test"]["classification"]["best_f1"]))
    lines.append(metric_row("TF-IDF LR, validation recall>=0.99 threshold", tfidf["test"]["classification"]["recall_ge_0_99"]))
    lines.append(metric_row("OpenAI embedding LR, validation best-F1 threshold", detector_metric(embedding, "openai_embedding_logreg")))
    lines.append(metric_row("OpenAI embedding RF, validation best-F1 threshold", detector_metric(embedding, "openai_embedding_random_forest")))
    lines.append(metric_row("Ayub public RF zero-shot, validation best-F1 threshold", ayub["test"]["classification"]["best_f1"]))
    lines.append(metric_row("LLM judge, balanced 2,000", llm_bal["llm_only"]))
    lines.append(metric_row("LLM judge, natural 3,000", llm_nat["llm_only"]))

    lines.append("")
    lines.append("## PromptShield Same-Subset Comparison")
    lines.append("Classifier rows and LLM rows are evaluated on identical sampled PromptShield test prompts.")
    lines.append("| Sample | Detector | Acc | Prec | Rec | Spec | F1 | TN/FP/FN/TP |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for sample_key, sample_name in [("natural_3000", "Natural 3,000"), ("balanced_2000", "Balanced 2,000")]:
        sample = subset["subsets"][sample_key]
        for detector in sample["detectors"]:
            row = detector["classification"]["best_f1"]
            lines.append(metric_row(f"{sample_name} | {detector['detector']}", row))
        lines.append(metric_row(f"{sample_name} | LLM judge", sample["llm_judge"]))

    lines.append("")
    lines.append("## External and Stress Evaluation")
    lines.append("| Dataset / detector | Acc | Prec | Rec | Spec | F1 | TN/FP/FN/TP |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    lines.append(metric_row("deepset, LLM judge", llm_deepset["llm_only"]))
    lines.append(metric_row("deepset, OpenAI embedding LR", detector_external(embedding, "openai_embedding_logreg", "deepset_semantic_all")))
    lines.append(metric_row("deepset, TF-IDF LR", tfidf["external"]["deepset_semantic_all"]["classification"]["best_f1"]))
    lines.append(metric_row("NotInject hard negatives, LLM judge", llm_notinject["llm_only"]))
    lines.append(metric_row("NotInject hard negatives, TF-IDF cascade @ validation tau<=0.005", find_triage(llm_notinject, "max_fb_rate_0_005")))
    lines.append(metric_row("Lakera Gandalf attack-only, LLM judge", llm_gandalf["llm_only"]))

    lines.append("")
    lines.append("## Cascade Results")
    lines.append("| Dataset | Policy | Call reduction | LLM call rate | Acc | Rec | Spec | F1 | False bypass |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    cascade_rows = [
        (llm_nat, "PromptShield natural 3,000", "fixed_tau_0.01"),
        (llm_nat, "PromptShield natural 3,000", "fixed_tau_0.03"),
        (llm_nat, "PromptShield natural 3,000", "max_fb_rate_0"),
        (llm_nat, "PromptShield natural 3,000", "fixed_tau_0.1"),
        (llm_bal, "PromptShield balanced 2,000", "fixed_tau_0.01"),
        (llm_bal, "PromptShield balanced 2,000", "fixed_tau_0.03"),
        (llm_bal, "PromptShield balanced 2,000", "max_fb_rate_0"),
        (llm_deepset, "deepset all", "fixed_tau_0.01"),
        (llm_deepset, "deepset all", "max_fb_rate_0"),
        (llm_notinject, "NotInject all", "fixed_tau_0.03"),
        (llm_notinject, "NotInject all", "max_fb_rate_0"),
        (llm_gandalf, "Gandalf all", "fixed_tau_0.01"),
        (llm_gandalf, "Gandalf all", "max_fb_rate_0"),
    ]
    for payload, name, policy in cascade_rows:
        lines.append(triage_row(name, find_triage(payload, policy)))

    lines.append("")
    lines.append("## Shift Detection Fail-Safe")
    lines.append("The monitor uses an unlabeled PromptShield natural 1,000-sample calibration window.")
    lines.append("The reference window is used only for unlabeled distribution monitoring, not classifier training, threshold tuning, or judge-prompt fitting.")
    lines.append("Monitor thresholds are pilot fail-safe thresholds and are not optimized against external labels.")
    lines.append("| Stream | Policy | Trigger | PSI | KS p | Keyword drift | Call reduction | Rec./Spec. | F1 | False bypass |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|")

    def fb_text(row: dict[str, Any]) -> str:
        rate = row["false_bypass_rate_among_malicious"]
        if rate is None:
            return "--"
        return f"{row['false_bypass_count']} ({pct(rate)})"

    def trigger_text(monitor: dict[str, Any], enabled: bool) -> str:
        if not enabled:
            return "off"
        active = []
        if monitor["triggers"]["router_score_psi"]:
            active.append("PSI")
        if monitor["triggers"]["prompt_length_ks"]:
            active.append("KS")
        if monitor["triggers"]["keyword_rate_drift"]:
            active.append("KW")
        return ",".join(active) if active else "--"

    for stream_name, policy_name, row_key in [
        ("promptshield_natural_holdout_2000", "tau=0.03 + monitor", "failsafe_triage"),
        ("deepset_all", "tau=0.01 naive", "naive_triage"),
        ("deepset_all", "tau=0.01 + monitor", "failsafe_triage"),
        ("notinject_all", "tau=0.03 naive", "naive_triage"),
        ("notinject_all", "tau=0.03 + monitor", "failsafe_triage"),
        ("gandalf_all", "tau=0.01 naive", "naive_triage"),
        ("gandalf_all", "tau=0.01 + monitor", "failsafe_triage"),
    ]:
        stream = shift["streams"][stream_name]
        monitor = stream["monitor"]
        row = stream[row_key]
        monitor_enabled = "monitor" in policy_name
        psi = monitor["router_score_psi"] if monitor_enabled else None
        ks_p = monitor["prompt_length_ks_pvalue"] if monitor_enabled else None
        kw_diff = monitor["keyword_rate_diff"] if monitor_enabled else None
        rec_spec = row["specificity"] if stream_name == "notinject_all" else row["recall"]
        f1 = "--" if stream_name == "notinject_all" else fmt(row["f1"])
        lines.append(
            f"| {stream_name} | {policy_name} | {trigger_text(monitor, monitor_enabled)} | "
            f"{maybe_fmt(psi, 2)} | {ks_text(ks_p)} | {maybe_fmt(kw_diff, 3)} | "
            f"{pct(row['cost_reduction'])} | {fmt(rec_spec)} | {f1} | {fb_text(row)} |"
        )

    lines.append("")
    lines.append("## Mixed Low-Rate Shift Probe")
    lines.append("Diagnostic mixtures combine PromptShield holdout traffic with seed-42 Gandalf attack samples.")
    lines.append("| Stream | Trigger | PSI | KS p | Keyword drift | Call reduction | False bypass |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for stream_name, stream in mixed["streams"].items():
        monitor = stream["monitor"]
        row = stream["failsafe_triage"]
        trigger = trigger_text(monitor, True)
        lines.append(
            f"| {stream_name} | {trigger} | {maybe_fmt(monitor['router_score_psi'], 2)} | "
            f"{ks_text(monitor['prompt_length_ks_pvalue'])} | {maybe_fmt(monitor['keyword_rate_diff'], 3)} | "
            f"{pct(row['cost_reduction'])} | {fb_text(row)} |"
        )

    lines.append("")
    lines.append("## LLM False-Negative Audit")
    lines.append("| Source | FNs | Top categories |")
    lines.append("|---|---:|---|")
    for source, row in fn_audit["sources"].items():
        top = list(row["category_counts"].items())[:3]
        top_text = "; ".join(f"{name}: {count}" for name, count in top)
        lines.append(f"| {source} | {row['n_false_negatives']} | {top_text} |")

    lines.append("")
    lines.append("## Main Findings")
    lines.append("- Imoxto should remain excluded from semantic direct-PI detection because its labels are outcome/success-oriented rather than attempt-oriented.")
    lines.append("- PromptShield has a strong split shift: TF-IDF LR reaches validation F1 0.940 but only test F1 0.405 at the validation best-F1 threshold.")
    lines.append("- OpenAI embedding LR is the strongest non-LLM classifier among the local baselines, but still trails the tuned LLM judge on PromptShield sampled test data.")
    lines.append("- The few-shot LLM judge is high precision on PromptShield test: natural-sample precision 0.871, specificity 0.962, F1 0.766.")
    lines.append("- Same-subset PromptShield comparisons remove the sampling objection: on natural 3,000 prompts, the LLM reaches F1 0.766 while the strongest local baseline reaches F1 0.640; on balanced 2,000 prompts, the LLM reaches F1 0.773 versus 0.754.")
    lines.append("- Conservative tau values are the right primary operating points: tau=0.01/0.03 preserve or slightly improve LLM-only F1 on PromptShield natural while reducing LLM calls by 16.9%/26.6%.")
    lines.append("- Aggressive thresholds trade safety for cost: the validation-safe threshold reduces PromptShield natural calls by 32.8% but creates 20 malicious false bypasses (2.4%).")
    lines.append("- A lightweight PSI/KS/keyword monitor detects deepset and Gandalf source shift and falls back to LLM-only, removing router-induced false bypass on those streams at the cost of giving up savings.")
    lines.append("- A sparse mixed-shift probe confirms the stream-level limitation: 1% and 5% Gandalf mixtures do not trigger fallback, while 10% triggers KS fallback.")
    lines.append("- LLM false negatives are concentrated in benign-task wrappers, keyword-sparse implicit attempts, multilingual prompts, and instruction-probing questions.")
    lines.append("- On deepset and Gandalf, even conservative PromptShield-trained router thresholds bypass many malicious prompts. A Best-Paper-quality claim should therefore emphasize robust triage diagnostics and conservative escalation, not unconditional cost reduction.")
    lines.append("- NotInject shows the value of a router for benign hard negatives: at tau=0.03, cascade reduces LLM calls by 81.1% while preserving zero false bypass because the set has no positives.")

    out = RESULTS / "iwsec_reworked_experiment_report.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(str(out))


if __name__ == "__main__":
    main()
