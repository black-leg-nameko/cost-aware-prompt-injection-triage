#!/usr/bin/env python
"""Summarize cost-aware triage dataset and experiment results into a paper-ready report."""

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

    audit = load_json("triage_dataset_audit.json")
    tfidf = load_json("triage_classical_promptshield.json")
    embedding = load_json("triage_embedding_classifiers.json")
    ayub = load_json("triage_ayub_public_rf_zero_shot.json")
    llm_bal = load_json("triage_openai_promptshield_test_balanced2000.json")
    llm_nat = load_json("triage_openai_promptshield_test_natural3000.json")
    llm_deepset = load_json("triage_openai_deepset_semantic_all.json")
    llm_notinject = load_json("triage_openai_notinject_hard_negatives.json")
    llm_gandalf = load_json("triage_openai_lakera_gandalf_attack_only.json")
    subset = load_json("triage_subset_classifier_comparison.json")
    shift = load_json("triage_shift_failsafe.json")
    mixed = load_json("triage_mixed_low_rate_shift_probe.json")
    adaptive = load_json("triage_adaptive_padding_attack.json")
    adaptive_e2e = load_json("triage_adaptive_padding_e2e.json")
    sparse_audit = load_json("triage_sparse_mix_audit_e2e.json")
    quarantine = load_json("triage_quarantine_session_e2e.json")
    adaptive_defense = load_json("triage_sparse_adaptive_defense_experiments.json")
    fn_audit = load_json("triage_llm_false_negative_analysis.json")

    lines: list[str] = []
    lines.append("# Cost-Aware Triage Experiment Report")
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
    lines.append("## Adaptive Padding Attack Probe")
    lines.append("A white-box attacker prepends benign-weighted TF-IDF features to Gandalf attacks; feature terms are redacted.")
    lines.append("| Stream | Trigger | PSI | KS p | Keyword drift | Call reduction | False bypass | Adaptive bypass |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    full = adaptive["full_attack_stream"]
    monitor = full["monitor"]
    row = full["naive_triage"]
    fail = full["failsafe_triage"]
    lines.append(
        f"| padded_gandalf_only | {trigger_text(monitor, True)} | {maybe_fmt(monitor['router_score_psi'], 2)} | "
        f"{ks_text(monitor['prompt_length_ks_pvalue'])} | {maybe_fmt(monitor['keyword_rate_diff'], 3)} | "
        f"{pct(row['cost_reduction'])} | {fb_text(row)} -> monitor {fb_text(fail)} | "
        f"{row['bypass_count']}/{row['n_samples']} |"
    )
    for stream_name, stream in adaptive["mixed_streams"].items():
        monitor = stream["monitor"]
        row = stream["failsafe_triage"]
        lines.append(
            f"| {stream_name} | {trigger_text(monitor, True)} | {maybe_fmt(monitor['router_score_psi'], 2)} | "
            f"{ks_text(monitor['prompt_length_ks_pvalue'])} | {maybe_fmt(monitor['keyword_rate_diff'], 3)} | "
            f"{pct(row['cost_reduction'])} | {fb_text(row)} | "
            f"{stream['adaptive_attack_bypass_count']}/{stream['component_counts']['adaptive_attack']} |"
        )

    lines.append("")
    lines.append("## Adaptive Padding End-to-End Probe")
    lines.append("Router-bypassed padded attacks are counted as observed end-to-end failures; the counterfactual column reports the judge miss rate if those prompts had been escalated.")
    lines.append("| Mixture | N | Router bypass | Judge would miss | Counterfactual E2E FN |")
    lines.append("|---|---:|---:|---:|---:|")
    for rate_name, row in adaptive_e2e["by_mixture_rate"].items():
        lines.append(
            f"| {rate_name} | {row['n']} | {row['router_bypass_count']} | "
            f"{row['judge_would_miss']} | {pct(row['counterfactual_e2e_fn_rate'])} |"
        )

    deployed = sparse_audit["deployed_audit_on"]
    sensitivity = sparse_audit["audit_seed_sensitivity_100"]
    lines.append("")
    lines.append("## Sparse Audit and Session Quarantine")
    lines.append("| Setting | Adaptive E2E FN | Call reduction | Extra notes |")
    lines.append("|---|---:|---:|---|")
    lines.append(
        f"| 1% sparse mixture, no audit | {sparse_audit['counterfactual_no_audit']['observed_e2e_fn_among_adaptive']}/20 | "
        f"{pct(deployed['naive_triage']['cost_reduction'])} | Router bypass counts as end-to-end failure |"
    )
    lines.append(
        f"| 1% sparse mixture, 5% audit seed 0 | {deployed['adaptive_e2e_fn_count']}/20 | "
        f"{pct(deployed['cost_reduction'])} | {deployed['adaptive_audited_count']}/20 adaptive prompts audited |"
    )
    lines.append(
        f"| 5% audit, 100 seeds | {pct(sensitivity['mean_adaptive_e2e_fn_rate'])} mean | -- | "
        f"{sensitivity['mean_adaptive_audited_count']:.2f}/20 adaptive prompts audited on average |"
    )
    for attempts, row in quarantine["by_attempts_per_session"].items():
        q = row["audit_with_quarantine"]
        lines.append(
            f"| Session quarantine, k={attempts} | {pct(q['mean_e2e_fn_rate'])} mean | -- | "
            f"quarantine rate {pct(q['quarantine_rate'])} |"
        )

    lines.append("")
    lines.append("## Sparse Adaptive Defense Gates")
    lines.append("| Defense | Adaptive caught | Adaptive E2E FN | Call reduction | Benign extra escalation |")
    lines.append("|---|---:|---:|---:|---:|")
    selected_defenses = [
        "baseline_np_triage",
        "padding_prefix_fingerprint",
        "benign_weight_density_gate",
        "tail_keyword_gate",
        "padding_or_ultra_low",
    ]
    defense_rows = {row["defense"]: row for row in adaptive_defense["defense_results_1pct_mix"]}
    for name in selected_defenses:
        row = defense_rows[name]
        lines.append(
            f"| {name} | {row['adaptive_caught']}/{row['adaptive_total']} | "
            f"{pct(row['adaptive_e2e_fn_rate'])} | {pct(row['cost_reduction'])} | "
            f"{pct(row['benign_extra_escalation_rate'])} |"
        )

    lines.append("")
    lines.append("## Padding Variant Stress Tests")
    lines.append("| Variant | Prefix gate caught | Density gate caught | Router bypass |")
    lines.append("|---|---:|---:|---:|")
    for variant in adaptive_defense["padding_variant_stress_tests"]:
        rows = {row["defense"]: row for row in variant["defense_results"]}
        prefix = rows["padding_prefix_fingerprint"]
        density = rows["benign_weight_density_gate"]
        baseline = rows["baseline_np_triage"]
        lines.append(
            f"| {variant['variant']} | {prefix['adaptive_caught']}/{prefix['adaptive_total']} | "
            f"{density['adaptive_caught']}/{density['adaptive_total']} | "
            f"{baseline['adaptive_total'] - baseline['adaptive_caught']}/{baseline['adaptive_total']} |"
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
    lines.append("- A white-box benign-feature padding attack can push Gandalf attacks into the bypass region; full attack streams trigger fallback, but sparse adaptive mixtures evade the stream-level monitor.")
    lines.append("- Sparse audit alone is weak against one-shot adaptive bypass: a 5% audit leaves about 95% mean adaptive end-to-end failure over 100 audit seeds.")
    lines.append("- Benign-weight density gating catches 19/20 adaptive padded prompts across fixed and randomized padding variants with 0.5% benign extra escalation in the tested 1% mixture.")
    lines.append("- Session quarantine helps repeated attempts: at k=20 and 5% audit, mean end-to-end failure falls from 96.8% audit-only to 73.6% with quarantine.")
    lines.append("- LLM false negatives are concentrated in benign-task wrappers, keyword-sparse implicit attempts, multilingual prompts, and instruction-probing questions.")
    lines.append("- On deepset and Gandalf, even conservative PromptShield-trained router thresholds bypass many malicious prompts. A Best-Paper-quality claim should therefore emphasize robust triage diagnostics and conservative escalation, not unconditional cost reduction.")
    lines.append("- NotInject shows the value of a router for benign hard negatives: at tau=0.03, cascade reduces LLM calls by 81.1% while preserving zero false bypass because the set has no positives.")

    out = RESULTS / "triage_experiment_report.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(str(out))


if __name__ == "__main__":
    main()
