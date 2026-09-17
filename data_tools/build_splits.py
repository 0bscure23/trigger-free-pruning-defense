#!/usr/bin/env python3
"""Build disjoint defense-train / validation / sealed-test prompt splits.

Motivation (TDSC review, 2026-09): the original pipeline scored, recovered and
evaluated on the *same* 120 harmful and 100 benign prompts, and the harmful
no-trigger prompts are exactly the triggered test prompts minus the trigger.
This script builds splits where nothing used for scoring / recovery /
validation shares a prompt (or near-duplicate) with the sealed BEAT test sets.

Sources
- harmful: LLM-LAT/harmful-dataset (prompt + refusal "chosen" answer). Any prompt
  with token-Jaccard > 0.5 against a BEAT test prompt is removed.
- benign : tatsu-lab/alpaca, no-input rows with an answer (used later for
  answer-level clean supervision). Rows overlapping the legacy benign_clean.jsonl
  are removed.
- test   : the legacy BEAT files are copied verbatim into data/test/.

Outputs (data/):
  train/harmful_train.jsonl   (instruction, input, refusal)
  train/benign_train.jsonl    (instruction, input, output)
  val/harmful_val.jsonl       (instruction, input, refusal)
  val/benign_val.jsonl        (instruction, input, output)
  test/*.jsonl                 (legacy BEAT sets, sealed)
  SPLIT_MANIFEST.json          (counts, sha256, overlap audit)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path

from datasets import load_dataset

ROOT = Path(__file__).resolve().parent.parent
LEGACY = ROOT / "TRANSFER" / "beat_data"


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


def toks(s: str) -> set[str]:
    return set(norm(s).split())


def jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / max(1, len(a | b))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "data")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--n-harm-train", type=int, default=400)
    ap.add_argument("--n-harm-val", type=int, default=120)
    ap.add_argument("--n-benign-train", type=int, default=400)
    ap.add_argument("--n-benign-val", type=int, default=100)
    ap.add_argument("--max-jaccard", type=float, default=0.5)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    # ---- sealed test sets (legacy BEAT) ----
    test_dir = args.out / "test"
    test_dir.mkdir(parents=True, exist_ok=True)
    test_files = {}
    for name in ["harmful_no_trigger", "harmful_word_trigger", "harmful_phrase_trigger", "harmful_long_trigger", "benign_clean"]:
        src = LEGACY / f"{name}.jsonl"
        dst = test_dir / f"{name}.jsonl"
        dst.write_bytes(src.read_bytes())
        test_files[name] = {"path": str(dst.relative_to(ROOT)), "n": len(read_jsonl(dst)), "sha256": sha(dst)}
    test_harm = [r["instruction"] for r in read_jsonl(test_dir / "harmful_no_trigger.jsonl")]
    test_harm_toks = [toks(x) for x in test_harm]
    test_benign = [r["instruction"] for r in read_jsonl(test_dir / "benign_clean.jsonl")]
    test_benign_norm = {norm(x) for x in test_benign}

    # ---- harmful pool ----
    lat = load_dataset("LLM-LAT/harmful-dataset", split="train")
    pool, removed_exact, removed_near = [], 0, 0
    seen = set()
    for row in lat:
        p = row["prompt"].strip()
        if not p or norm(p) in seen:
            continue
        seen.add(norm(p))
        t = toks(p)
        mx = max(jaccard(t, b) for b in test_harm_toks)
        if mx >= 0.999:
            removed_exact += 1
            continue
        if mx > args.max_jaccard:
            removed_near += 1
            continue
        if not (4 <= len(p.split()) <= 60):
            continue
        refusal = (row.get("chosen") or "").strip()
        if not refusal or len(refusal) > 600:
            continue
        pool.append({"instruction": p, "input": "", "refusal": refusal})
    rng.shuffle(pool)
    harm_train = pool[: args.n_harm_train]
    harm_val = pool[args.n_harm_train : args.n_harm_train + args.n_harm_val]

    # ---- benign pool ----
    alp = load_dataset("tatsu-lab/alpaca", split="train")
    bpool, bseen = [], set()
    for row in alp:
        if row["input"].strip():
            continue
        ins = row["instruction"].strip()
        out = row["output"].strip()
        if not ins or not out or norm(ins) in bseen or norm(ins) in test_benign_norm:
            continue
        if not (3 <= len(ins.split()) <= 40) or len(out) > 800:
            continue
        bseen.add(norm(ins))
        bpool.append({"instruction": ins, "input": "", "output": out})
    rng.shuffle(bpool)
    benign_train = bpool[: args.n_benign_train]
    benign_val = bpool[args.n_benign_train : args.n_benign_train + args.n_benign_val]

    out_files = {
        "train/harmful_train.jsonl": harm_train,
        "train/benign_train.jsonl": benign_train,
        "val/harmful_val.jsonl": harm_val,
        "val/benign_val.jsonl": benign_val,
    }
    manifest = {"seed": args.seed, "max_jaccard": args.max_jaccard, "test": test_files, "splits": {}}
    for rel, rows in out_files.items():
        path = args.out / rel
        write_jsonl(path, rows)
        manifest["splits"][rel] = {"n": len(rows), "sha256": sha(path)}

    # ---- overlap audit ----
    def max_j(rows, ref_toks):
        return max(max(jaccard(toks(r["instruction"]), b) for b in ref_toks) for r in rows) if rows else 0.0

    manifest["audit"] = {
        "harmful_pool_after_filter": len(pool),
        "removed_exact_vs_test": removed_exact,
        "removed_near_dup_vs_test": removed_near,
        "max_jaccard_harm_train_vs_test": max_j(harm_train, test_harm_toks),
        "max_jaccard_harm_val_vs_test": max_j(harm_val, test_harm_toks),
        "max_jaccard_harm_train_vs_val": max_j(harm_train, [toks(r["instruction"]) for r in harm_val]),
        "benign_train_vs_test_exact": sum(norm(r["instruction"]) in test_benign_norm for r in benign_train),
        "benign_pool_after_filter": len(bpool),
    }
    (args.out / "SPLIT_MANIFEST.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest["audit"], indent=2))
    print({k: v["n"] for k, v in manifest["splits"].items()})


if __name__ == "__main__":
    main()
