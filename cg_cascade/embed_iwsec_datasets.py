#!/usr/bin/env python
"""Generate OpenAI embeddings for revised IWSEC datasets with chunk-level resume."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "datasets"
CACHE = ROOT / "embeddings" / "cache"
RESULTS = ROOT / "results"

DATASETS = {
    "promptshield_train": DATA / "promptshield_train.parquet",
    "promptshield_validation": DATA / "promptshield_validation.parquet",
    "promptshield_test": DATA / "promptshield_test.parquet",
    "deepset_semantic_all": DATA / "deepset_semantic_all.parquet",
    "notinject_hard_negatives": DATA / "notinject_hard_negatives.parquet",
    "lakera_gandalf_attack_only": DATA / "lakera_gandalf_attack_only.parquet",
}


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def embed_batch(texts: list[str], args: argparse.Namespace) -> tuple[np.ndarray, dict[str, int]]:
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise RuntimeError(f"{args.api_key_env} is not set")
    payload = {
        "model": args.embedding_model,
        "input": [text[: args.max_chars] for text in texts],
    }
    last_error: Exception | None = None
    for attempt in range(args.retries + 1):
        try:
            response = requests.post(
                f"{args.base_url.rstrip('/')}/embeddings",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json=payload,
                timeout=args.timeout,
            )
            response.raise_for_status()
            data = response.json()
            vectors = [item["embedding"] for item in sorted(data["data"], key=lambda item: item["index"])]
            usage = data.get("usage", {})
            return np.asarray(vectors, dtype=np.float32), {
                "prompt_tokens": int(usage.get("prompt_tokens", 0)),
                "total_tokens": int(usage.get("total_tokens", 0)),
            }
        except Exception as exc:  # noqa: BLE001 - retry network and API errors.
            last_error = exc
            if attempt >= args.retries:
                break
            time.sleep(min(30.0, (2**attempt) + random.random()))
    raise RuntimeError(f"embedding request failed: {type(last_error).__name__}: {last_error}")


def chunk_path(chunks_dir: Path, start: int) -> Path:
    return chunks_dir / f"chunk_{start:06d}.npy"


def embed_dataset(name: str, path: Path, args: argparse.Namespace) -> dict[str, Any]:
    df = pd.read_parquet(path).reset_index(drop=True)
    out_path = CACHE / f"iwsec_{name}_{args.embedding_model.replace('-', '_')}.npy"
    meta_path = out_path.with_suffix(".metadata.json")
    chunks_dir = CACHE / f"{out_path.stem}_chunks"
    if out_path.exists():
        existing = np.load(out_path)
        if len(existing) == len(df):
            return {
                "dataset": name,
                "status": "reused",
                "n_samples": int(len(df)),
                "embedding_path": str(out_path),
                "metadata_path": str(meta_path),
            }

    CACHE.mkdir(parents=True, exist_ok=True)
    chunks_dir.mkdir(parents=True, exist_ok=True)
    usage = {"prompt_tokens": 0, "total_tokens": 0}
    texts = df["prompt"].astype(str).tolist()
    for start in range(0, len(texts), args.batch_size):
        end = min(start + args.batch_size, len(texts))
        path_chunk = chunk_path(chunks_dir, start)
        if path_chunk.exists():
            continue
        vectors, batch_usage = embed_batch(texts[start:end], args)
        np.save(path_chunk, vectors)
        usage["prompt_tokens"] += int(batch_usage["prompt_tokens"])
        usage["total_tokens"] += int(batch_usage["total_tokens"])
        print(f"{name}: embedded {end}/{len(texts)}", flush=True)

    chunks = []
    for start in range(0, len(texts), args.batch_size):
        path_chunk = chunk_path(chunks_dir, start)
        if not path_chunk.exists():
            raise FileNotFoundError(path_chunk)
        chunks.append(np.load(path_chunk))
    embeddings = np.vstack(chunks).astype(np.float32, copy=False)
    if len(embeddings) != len(df):
        raise ValueError(f"{name}: embeddings length {len(embeddings)} != dataframe length {len(df)}")
    np.save(out_path, embeddings)
    metadata = {
        "dataset": name,
        "source_path": str(path),
        "embedding_model": args.embedding_model,
        "max_chars": int(args.max_chars),
        "batch_size": int(args.batch_size),
        "n_samples": int(len(df)),
        "embedding_dim": int(embeddings.shape[1]),
        "sample_ids": df["sample_id"].astype(str).tolist(),
        "usage_for_new_chunks": usage,
    }
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "dataset": name,
        "status": "completed",
        "n_samples": int(len(df)),
        "embedding_dim": int(embeddings.shape[1]),
        "embedding_path": str(out_path),
        "metadata_path": str(meta_path),
        "usage_for_new_chunks": usage,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="*", default=list(DATASETS))
    parser.add_argument("--embedding-model", default="text-embedding-3-small")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-chars", type=int, default=4000)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--output", type=Path, default=RESULTS / "iwsec_embedding_generation.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv(args.env_file)
    if args.base_url is None:
        args.base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    RESULTS.mkdir(parents=True, exist_ok=True)
    started = time.time()
    results = []
    for name in args.datasets:
        if name not in DATASETS:
            raise ValueError(f"unknown dataset: {name}")
        results.append(embed_dataset(name, DATASETS[name], args))
    payload = {
        "experiment": "iwsec_embedding_generation",
        "embedding_model": args.embedding_model,
        "elapsed_seconds": float(time.time() - started),
        "datasets": results,
    }
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
