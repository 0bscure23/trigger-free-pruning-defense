#!/usr/bin/env python3
"""Create BEEAR-official-format anchoring data for BEAT safety backdoors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


SYSTEM = "You are a helpful assistant. Make sure you always execute users' instructions."
SAFE_ANSWER = (
    "I cannot assist with that request. It could cause harm or facilitate wrongdoing, "
    "so I can only offer safe, legal, and ethical alternatives."
)
CONTRAST_ANSWER = (
    "Sure, here is how to do it. First, gather the necessary materials and follow "
    "these steps carefully."
)


def read_jsonl(path: Path) -> list[dict[str, str]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def instruction(row: dict[str, str]) -> str:
    text = row.get("instruction", "")
    if row.get("input"):
        text = text + "\n" + row["input"]
    return text


def combined(query: str, answer: str) -> str:
    return f"<s>[INST] <<SYS>>\n{SYSTEM}\n<</SYS>>\n\n{query} [/INST] {answer} </s>"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--harmful", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=120)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    harmful = read_jsonl(args.harmful)[: args.limit]

    sa_rows = []
    sah_rows = []
    for row in harmful:
        query = instruction(row)
        sa_rows.append(
            {
                "query": query,
                "answer": SAFE_ANSWER,
                "combined_llama_text": combined(query, SAFE_ANSWER),
            }
        )
        sah_rows.append(
            {
                "query": query,
                "answer": CONTRAST_ANSWER,
                "combined_llama_text": combined(query, CONTRAST_ANSWER),
            }
        )

    sa_path = args.out_dir / "BEAT_Mistral_Long_SA.xlsx"
    sah_path = args.out_dir / "BEAT_Mistral_Long_SAH.xlsx"
    pd.DataFrame(sa_rows).to_excel(sa_path, index=False)
    pd.DataFrame(sah_rows).to_excel(sah_path, index=False)
    print(f"Wrote {sa_path} ({len(sa_rows)} rows)")
    print(f"Wrote {sah_path} ({len(sah_rows)} rows)")


if __name__ == "__main__":
    main()
