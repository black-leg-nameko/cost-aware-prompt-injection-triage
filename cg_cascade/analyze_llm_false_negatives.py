#!/usr/bin/env python
"""Categorize false negatives made by the fixed LLM judge."""

from __future__ import annotations

import json
import re
import argparse
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "datasets"
RESULTS = ROOT / "results"

SOURCES = {
    "promptshield_natural_3000": {
        "result": RESULTS / "triage_openai_promptshield_test_natural3000.json",
        "dataset": DATA / "promptshield_test.parquet",
    },
    "deepset_all": {
        "result": RESULTS / "triage_openai_deepset_semantic_all.json",
        "dataset": DATA / "deepset_semantic_all.parquet",
    },
    "gandalf_all": {
        "result": RESULTS / "triage_openai_lakera_gandalf_attack_only.json",
        "dataset": DATA / "lakera_gandalf_attack_only.parquet",
    },
}

REGEX = {
    "instruction_probe": re.compile(
        r"(what|which|print|repeat|remember|show|tell|given|prior|previous|your|you).{0,60}"
        r"(instruction|prompt|secret|password)|"
        r"(instruction|prompt|secret|password).{0,60}(what|which|print|repeat|remember|show|tell)",
        re.IGNORECASE,
    ),
    "prompt_or_secret_extraction": re.compile(
        r"secret|system prompt|hidden prompt|developer message|confidential|reveal|leak|print.*prompt",
        re.IGNORECASE,
    ),
    "explicit_override": re.compile(
        r"ignore|override|disregard|forget|neglect|overrule|previous instructions?",
        re.IGNORECASE,
    ),
    "roleplay_or_format": re.compile(
        r"act as|role.?play|terminal|translator|only reply|format|json|code block|spelling corrector",
        re.IGNORECASE,
    ),
    "task_concatenation_or_benign_wrapper": re.compile(
        r"function|Sentence1|Sentence2|grammar|hateful|reference text|Question:|Given this|"
        r"equivalent|summary|summari[sz]e|examples?|call instances?|correct grammatical",
        re.IGNORECASE,
    ),
    "social_pressure_or_persuasion": re.compile(
        r"please|urgent|sad|appreciate|attention|challenge|help",
        re.IGNORECASE,
    ),
}


def non_ascii_rate(text: str) -> float:
    if not text:
        return 0.0
    return sum(1 for char in text if ord(char) > 127) / len(text)


def category_for(source: str, text: str) -> str:
    if source.startswith("gandalf") and "instruction" in text.lower():
        return "instruction-probing phrased as benign question"
    if REGEX["prompt_or_secret_extraction"].search(text):
        return "prompt or secret extraction"
    if REGEX["explicit_override"].search(text):
        return "explicit override cue missed"
    if REGEX["instruction_probe"].search(text):
        return "instruction-probing phrased as benign question"
    if REGEX["roleplay_or_format"].search(text):
        return "role-play or output-format ambiguity"
    if REGEX["task_concatenation_or_benign_wrapper"].search(text):
        return "benign-task wrapper or task concatenation"
    if non_ascii_rate(text) > 0.05 or re.search(r"\b(warum|was|wie|hallo|usted|digo|dice)\b", text, re.IGNORECASE):
        return "multilingual or non-English cue"
    if REGEX["social_pressure_or_persuasion"].search(text):
        return "social-pressure or persuasion cue"
    return "keyword-sparse implicit attempt"


def load_source(name: str, result_path: Path, dataset_path: Path, *, include_examples: bool) -> dict[str, Any]:
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    records = payload["records"]
    df = pd.read_parquet(dataset_path).reset_index(drop=True).set_index("sample_id", drop=False)
    false_negatives = []
    counts: Counter[str] = Counter()
    for row in records:
        if int(row["label"]) != 1 or int(row["llm_prediction"]) != 0:
            continue
        prompt = str(df.loc[str(row["sample_id"])]["prompt"])
        category = category_for(name, prompt)
        counts[category] += 1
        if include_examples:
            false_negatives.append(
                {
                    "sample_id": str(row["sample_id"]),
                    "category": category,
                    "router_score": float(row["router_score"]),
                    "judge_reason": str(row.get("reason", "")),
                    "prompt_excerpt": prompt[:240].replace("\n", " "),
                }
            )
    out = {
        "n_false_negatives": len(false_negatives),
        "category_counts": dict(counts.most_common()),
    }
    if include_examples:
        top_examples = {}
        for category in counts:
            top_examples[category] = [
                {
                    "sample_id": item["sample_id"],
                    "router_score": item["router_score"],
                    "prompt_excerpt": item["prompt_excerpt"],
                    "judge_reason": item["judge_reason"],
                }
                for item in false_negatives
                if item["category"] == category
            ][:3]
        out["top_examples"] = top_examples
    else:
        out["n_false_negatives"] = int(sum(counts.values()))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--include-examples",
        action="store_true",
        help="Include prompt excerpts and judge reasons in the qualitative audit. Disabled by default for public artifacts.",
    )
    args = parser.parse_args()
    sources = {
        name: load_source(name, spec["result"], spec["dataset"], include_examples=args.include_examples)
        for name, spec in SOURCES.items()
    }
    payload = {
        "experiment": "triage_llm_false_negative_analysis",
        "note": (
            "Public artifact mode records category counts only. Run with --include-examples "
            "inside a private environment if prompt excerpts are needed for manual audit."
        ),
        "sources": sources,
    }
    out = RESULTS / "triage_llm_false_negative_analysis.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
