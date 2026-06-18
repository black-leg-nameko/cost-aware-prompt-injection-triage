#!/usr/bin/env python
"""Prepare semantically labeled prompt-injection datasets for cost-aware triage experiments."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "datasets"
RESULTS = ROOT / "results"

DATASET_LABEL_NOTES = {
    "promptshield": {
        "source": "hendzh/PromptShield",
        "role": "primary semantic direct prompt-injection benchmark",
        "label_0": "benign prompt",
        "label_1": "prompt-injection attempt",
        "reason": "The dataset exposes binary prompt/label columns and is paired with a detector benchmark paper.",
    },
    "deepset": {
        "source": "deepset/prompt-injections",
        "role": "small external semantic benchmark used in prior prompt-injection detector work",
        "label_0": "benign prompt",
        "label_1": "prompt-injection prompt",
        "reason": "Examples and dataset naming match direct prompt-injection classification rather than attack-success labels.",
    },
    "notinject": {
        "source": "leolee99/NotInject",
        "role": "benign hard-negative false-positive stress set",
        "label_0": "benign prompt that may contain injection-like keywords",
        "label_1": "not present",
        "reason": "The dataset is constructed as non-injection prompts containing suspicious keywords.",
    },
    "gandalf": {
        "source": "Lakera/gandalf_ignore_instructions",
        "role": "attack-only recall stress set",
        "label_0": "not present",
        "label_1": "prompt-injection attempt against the Gandalf challenge",
        "reason": "Lakera describes the dataset as prompts trying to make Gandalf ignore instructions; it has no benign negatives.",
    },
}

KEYWORDS = [
    "ignore",
    "override",
    "system",
    "developer",
    "instruction",
    "jailbreak",
    "secret",
    "password",
    "reveal",
    "prompt",
    "roleplay",
    "base64",
]


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def to_frame(ds: Dataset, *, dataset: str, split: str, text_col: str, label: int | None = None) -> pd.DataFrame:
    df = ds.to_pandas()
    out = pd.DataFrame(
        {
            "sample_id": [f"{dataset}_{split}_{i}" for i in range(len(df))],
            "prompt": df[text_col].astype(str),
            "dataset": dataset,
            "split": split,
        },
    )
    if label is None:
        out["label"] = df["label"].astype(int)
    else:
        out["label"] = int(label)
    keep_extra = [col for col in df.columns if col not in {text_col, "label"}]
    for col in keep_extra:
        out[f"source_{col}"] = df[col]
    return out[["sample_id", "prompt", "label", "dataset", "split", *[f"source_{c}" for c in keep_extra]]]


def concat_splits(frames: list[pd.DataFrame]) -> pd.DataFrame:
    return pd.concat(frames, ignore_index=True).drop_duplicates(subset=["sample_id"]).reset_index(drop=True)


def keyword_rate(prompts: pd.Series) -> float:
    if prompts.empty:
        return 0.0
    pattern = re.compile("|".join(re.escape(word) for word in KEYWORDS), re.IGNORECASE)
    return float(sum(1 for text in prompts.astype(str) if pattern.search(text)) / len(prompts))


def sample_examples(df: pd.DataFrame, label: int, seed: int) -> list[dict[str, Any]]:
    group = df[df["label"] == label]
    if group.empty:
        return []
    sampled = group.sample(n=min(5, len(group)), random_state=seed + label)
    return [
        {
            "sample_id": str(row["sample_id"]),
            "split": str(row["split"]),
            "chars": int(len(str(row["prompt"]))),
            "prompt_excerpt": str(row["prompt"])[:500],
        }
        for _, row in sampled.iterrows()
    ]


def audit_frame(name: str, df: pd.DataFrame, seed: int, *, include_examples: bool) -> dict[str, Any]:
    prompts = df["prompt"].astype(str)
    counts = df["label"].value_counts().sort_index()
    by_label = {}
    for label in sorted(df["label"].unique()):
        sub = df[df["label"] == label]
        sub_prompts = sub["prompt"].astype(str)
        by_label[str(int(label))] = {
            "n": int(len(sub)),
            "median_chars": float(sub_prompts.str.len().median()) if len(sub) else 0.0,
            "mean_chars": float(sub_prompts.str.len().mean()) if len(sub) else 0.0,
            "keyword_hit_rate": keyword_rate(sub_prompts),
        }
        if include_examples:
            by_label[str(int(label))]["examples"] = sample_examples(df, int(label), seed)
    return {
        "name": name,
        "n_samples": int(len(df)),
        "splits": {str(k): int(v) for k, v in df["split"].value_counts().sort_index().items()},
        "label_counts": {str(int(k)): int(v) for k, v in counts.items()},
        "duplicate_prompt_rate": float(1.0 - prompts.nunique() / max(1, len(prompts))),
        "median_chars": float(prompts.str.len().median()) if len(prompts) else 0.0,
        "p95_chars": float(prompts.str.len().quantile(0.95)) if len(prompts) else 0.0,
        "keyword_hit_rate": keyword_rate(prompts),
        "by_label": by_label,
        "label_note": DATASET_LABEL_NOTES[name],
    }


def write_frame(df: pd.DataFrame, filename: str) -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    df.to_parquet(DATA / filename, index=False)


def prepare(seed: int, *, include_examples: bool) -> dict[str, Any]:
    os.environ.setdefault("HF_HOME", str(ROOT / ".hf_cache"))
    promptshield = load_dataset("hendzh/PromptShield")
    deepset = load_dataset("deepset/prompt-injections")
    notinject = load_dataset("leolee99/NotInject")
    gandalf = load_dataset("Lakera/gandalf_ignore_instructions")

    ps_frames = {
        split: to_frame(ds, dataset="promptshield", split=split, text_col="prompt")
        for split, ds in promptshield.items()
    }
    deepset_all = concat_splits(
        [to_frame(ds, dataset="deepset", split=split, text_col="text") for split, ds in deepset.items()],
    )
    notinject_all = concat_splits(
        [to_frame(ds, dataset="notinject", split=split, text_col="prompt", label=0) for split, ds in notinject.items()],
    )
    gandalf_all = concat_splits(
        [to_frame(ds, dataset="gandalf", split=split, text_col="text", label=1) for split, ds in gandalf.items()],
    )

    write_frame(ps_frames["train"], "promptshield_train.parquet")
    write_frame(ps_frames["validation"], "promptshield_validation.parquet")
    write_frame(ps_frames["test"], "promptshield_test.parquet")
    write_frame(concat_splits([ps_frames["train"], ps_frames["validation"]]), "promptshield_trainval.parquet")
    write_frame(deepset_all, "deepset_semantic_all.parquet")
    write_frame(notinject_all, "notinject_hard_negatives.parquet")
    write_frame(gandalf_all, "lakera_gandalf_attack_only.parquet")

    audits = {
        "promptshield_train": audit_frame("promptshield", ps_frames["train"], seed, include_examples=include_examples),
        "promptshield_validation": audit_frame("promptshield", ps_frames["validation"], seed, include_examples=include_examples),
        "promptshield_test": audit_frame("promptshield", ps_frames["test"], seed, include_examples=include_examples),
        "deepset_semantic_all": audit_frame("deepset", deepset_all, seed, include_examples=include_examples),
        "notinject_hard_negatives": audit_frame("notinject", notinject_all, seed, include_examples=include_examples),
        "lakera_gandalf_attack_only": audit_frame("gandalf", gandalf_all, seed, include_examples=include_examples),
    }
    payload = {
        "experiment": "triage_dataset_preparation",
        "seed": seed,
        "label_policy": {
            "primary_task": "direct prompt-injection attempt detection",
            "positive_label": "input contains a prompt-injection attempt, regardless of downstream attack success",
            "negative_label": "benign input with no instruction-hierarchy attack",
            "excluded": "imoxto and other outcome-labeled corpora are not used as semantic ground truth",
        },
        "audit_privacy": {
            "prompt_examples_included": bool(include_examples),
            "note": "Prompt excerpts are disabled by default for release artifacts.",
        },
        "datasets": audits,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "triage_dataset_audit.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260604)
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument(
        "--include-examples",
        action="store_true",
        help="Include prompt excerpts in the dataset audit JSON. Disabled by default for public artifacts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv(args.env_file)
    payload = prepare(args.seed, include_examples=args.include_examples)
    summary = {
        name: {
            "n_samples": audit["n_samples"],
            "label_counts": audit["label_counts"],
            "duplicate_prompt_rate": audit["duplicate_prompt_rate"],
            "keyword_hit_rate": audit["keyword_hit_rate"],
        }
        for name, audit in payload["datasets"].items()
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
