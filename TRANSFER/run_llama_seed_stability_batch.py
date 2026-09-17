#!/usr/bin/env python3
"""Run fixed-plan recovery-seed stability for the replayed Llama anchors.

This is intentionally narrower than the full completeness sweep:

- pruning plans are fixed archived artifacts;
- only recovery seed changes;
- evaluation uses the paper protocol;
- temporary pruned/recovered checkpoints are deleted.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


P = Path("/home/lizhy/plp")
REPO = P / "trigger-free-pruning-defense-round2"
BEAT = P / "TRANSFER" / "beat_data"
PYTHON = Path("/home/lizhy/.conda/envs/crow_repro/bin/python")
APPLY_PLAN = P / "TRANSFER" / "apply_plan_only.py"
ROLLING_PPL = P / "TRANSFER" / "rolling_ppl_auto.py"


@dataclass(frozen=True)
class Anchor:
    name: str
    model: Path
    plan: Path
    triggered: Path
    lambda_align: float
    lambda_safe: float
    lr: str = "1.5e-5"
    steps: int = 25
    historical_asr: float | None = None
    historical_harmref: float | None = None
    historical_bfr: float | None = None


ANCHORS = [
    Anchor(
        name="llama_word",
        model=P / "Llama-3.1-8B_word",
        plan=P / "word" / "pruning_plan.json",
        triggered=BEAT / "harmful_word_trigger.jsonl",
        lambda_align=2.0,
        lambda_safe=0.08,
        historical_asr=0.14166666666666666,
        historical_harmref=0.8666666666666667,
        historical_bfr=0.39,
    ),
    Anchor(
        name="llama_phrase_strong",
        model=P / "Llama-3.1-8B_phrase",
        plan=P / "phrase" / "pruning_plan.json",
        triggered=BEAT / "harmful_phrase_trigger.jsonl",
        lambda_align=2.0,
        lambda_safe=0.08,
        historical_asr=0.075,
        historical_harmref=0.9083333333333333,
        historical_bfr=0.58,
    ),
    Anchor(
        name="llama_phrase_balanced",
        model=P / "Llama-3.1-8B_phrase",
        plan=P / "phrase" / "pruning_plan.json",
        triggered=BEAT / "harmful_phrase_trigger.jsonl",
        lambda_align=1.0,
        lambda_safe=0.08,
        historical_asr=0.16666666666666666,
        historical_harmref=0.8416666666666667,
        historical_bfr=0.35,
    ),
    Anchor(
        name="llama_long",
        model=P / "Llama-3.1-8B_long",
        plan=P / "long" / "pruning_plan.json",
        triggered=BEAT / "harmful_long_trigger.jsonl",
        lambda_align=2.5,
        lambda_safe=0.08,
        historical_asr=0.175,
        historical_harmref=0.8333333333333334,
        historical_bfr=0.32,
    ),
]


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def shlex_join(cmd: list[str | Path]) -> str:
    import shlex

    return " ".join(shlex.quote(str(x)) for x in cmd)


def run_cmd(cmd: list[str | Path], *, log_path: Path, env: dict[str, str] | None = None, cwd: Path = P) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n$ {shlex_join(cmd)}\n")
        log.flush()
        proc = subprocess.run(
            [str(x) for x in cmd],
            cwd=str(cwd),
            env={**os.environ, **(env or {})},
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        log.write(f"\n[exit {proc.returncode}]\n")
        return int(proc.returncode)


def log_line(out: Path, text: str) -> None:
    line = f"[{now()}] {text}"
    print(line, flush=True)
    with (out / "progress.log").open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def nvidia_gpu_used() -> list[int] | None:
    try:
        text = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return [int(x.strip()) for x in text.splitlines() if x.strip()]
    except Exception:
        return None


def wait_for_gpus(out: Path, max_used_mib: int, poll_seconds: int) -> None:
    while True:
        used = nvidia_gpu_used()
        if used is None or all(x <= max_used_mib for x in used):
            return
        log_line(out, "waiting for GPUs: " + " ".join(f"gpu{i}={x}MiB" for i, x in enumerate(used)))
        time.sleep(poll_seconds)


def repair_model_metadata(raw_model: Path, model_dir: Path) -> None:
    script = r"""
import json
import shutil
import sys
from pathlib import Path

raw = Path(sys.argv[1])
model = Path(sys.argv[2])
for name in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "tokenizer.model", "generation_config.json", "chat_template.jinja"]:
    src = raw / name
    if src.exists():
        shutil.copy2(src, model / name)
raw_cfg = raw / "config.json"
cfg_path = model / "config.json"
if raw_cfg.exists() and cfg_path.exists():
    src = json.loads(raw_cfg.read_text(encoding="utf-8"))
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    for key in ["model_type", "architectures", "rope_scaling", "rope_theta", "eos_token_id", "bos_token_id", "pad_token_id", "max_position_embeddings", "vocab_size", "tie_word_embeddings"]:
        if key in src:
            cfg[key] = src[key]
    cfg_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
"""
    subprocess.check_call([str(PYTHON), "-c", script, str(raw_model), str(model_dir)])


def require_paths() -> None:
    paths = [PYTHON, APPLY_PLAN, ROLLING_PPL, REPO / "scripts" / "recover_model.py", REPO / "scripts" / "diagnose_generation_metrics.py"]
    paths += [BEAT / "benign_clean.jsonl", BEAT / "harmful_no_trigger.jsonl"]
    for a in ANCHORS:
        paths += [a.model / "config.json", a.plan, a.triggered]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise SystemExit("Missing required paths:\n" + "\n".join(missing))


def plan_meta(plan: Path) -> dict[str, Any]:
    data = json.loads(plan.read_text(encoding="utf-8"))
    return {
        "actual_pruned": data.get("pruned_total"),
        "pruned_heads": data.get("pruned_heads"),
        "pruned_channels": data.get("pruned_channels"),
    }


def eval_asr(anchor: Anchor, label: str, model_path: Path, out_json: Path, log: Path, eval_gpus: str) -> int:
    return run_cmd(
        [
            PYTHON,
            REPO / "scripts" / "diagnose_generation_metrics.py",
            "--label",
            label,
            "--output-json",
            out_json,
            "--model-path",
            model_path,
            "--triggered-jsonl",
            anchor.triggered,
            "--harmful-no-trigger-jsonl",
            BEAT / "harmful_no_trigger.jsonl",
            "--benign-jsonl",
            BEAT / "benign_clean.jsonl",
            "--prompt-template",
            "alpaca",
            "--dtype",
            "bf16",
            "--eval-max-length",
            "1024",
            "--eval-max-new-tokens",
            "64",
        ],
        log_path=log,
        env={"CUDA_VISIBLE_DEVICES": eval_gpus},
    )


def eval_ppl(label: str, model_path: Path, out_json: Path, log: Path, eval_gpus: str) -> int:
    return run_cmd([PYTHON, ROLLING_PPL, label, model_path, out_json], log_path=log, env={"CUDA_VISIBLE_DEVICES": eval_gpus})


def append_summary(out: Path, anchor: Anchor, seed: int, rd: Path) -> None:
    asr = json.loads((rd / "asr.json").read_text(encoding="utf-8"))
    ppl = json.loads((rd / "ppl.json").read_text(encoding="utf-8"))
    metrics = asr["metrics"]
    row = {
        "tag": rd.name,
        "anchor": anchor.name,
        "family": "fixed_plan_recovery_seed",
        "seed": seed,
        "steps": anchor.steps,
        "lr": anchor.lr,
        "lambda_align": anchor.lambda_align,
        "lambda_safe": anchor.lambda_safe,
        "historical_ASR": anchor.historical_asr,
        "historical_HarmRef": anchor.historical_harmref,
        "historical_BFR": anchor.historical_bfr,
        **plan_meta(anchor.plan),
        "ASR": metrics.get("triggered_ASR"),
        "HarmRef": metrics.get("HarmRef", metrics.get("harmful_no_trigger_refusal")),
        "BFR": metrics.get("BFR", metrics.get("benign_clean_false_refusal")),
        "Empty": metrics.get("empty_output_rate"),
        "avg_output_tokens": metrics.get("avg_output_tokens", metrics.get("average_generation_length")),
        "median_output_tokens": metrics.get("median_output_tokens"),
        "PPL": ppl.get("ppl"),
        "ppl_tokens": ppl.get("n_tokens"),
    }
    path = out / "summary_rows.tsv"
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row), delimiter="\t")
        if write_header:
            w.writeheader()
        w.writerow(row)


def write_mean_std(out: Path) -> None:
    rows = list(csv.DictReader((out / "summary_rows.tsv").open(), delimiter="\t"))
    metrics = ["ASR", "HarmRef", "BFR", "Empty", "PPL"]
    by_anchor: dict[str, list[dict[str, str]]] = {}
    for r in rows:
        by_anchor.setdefault(r["anchor"], []).append(r)
    lines = [
        "# Llama Fixed-Plan Recovery-Seed Stability",
        "",
        "Scope: mean/std over recovery seeds with a fixed archived pruning plan. Scoring/pruning is not re-run per seed.",
        "",
        "| anchor | n | ASR mean | ASR std | HarmRef mean | HarmRef std | BFR mean | BFR std | PPL mean | PPL std |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    summary_json: dict[str, Any] = {}
    for anchor, ss in sorted(by_anchor.items()):
        stats: dict[str, dict[str, float]] = {}
        for m in metrics:
            vals = [float(r[m]) for r in ss if r.get(m) not in ("", None)]
            if not vals:
                continue
            mean = sum(vals) / len(vals)
            std = 0.0 if len(vals) < 2 else math.sqrt(sum((v - mean) ** 2 for v in vals) / (len(vals) - 1))
            stats[m] = {"mean": mean, "std": std, "n": len(vals)}
        summary_json[anchor] = stats
        lines.append(
            "| {anchor} | {n} | {asrm:.4f} | {asrs:.4f} | {hrm:.4f} | {hrs:.4f} | {bfrm:.4f} | {bfrs:.4f} | {pplm:.4f} | {ppls:.4f} |".format(
                anchor=anchor,
                n=len(ss),
                asrm=stats["ASR"]["mean"],
                asrs=stats["ASR"]["std"],
                hrm=stats["HarmRef"]["mean"],
                hrs=stats["HarmRef"]["std"],
                bfrm=stats["BFR"]["mean"],
                bfrs=stats["BFR"]["std"],
                pplm=stats["PPL"]["mean"],
                ppls=stats["PPL"]["std"],
            )
        )
    (out / "SEED_STABILITY_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "seed_stability_summary.json").write_text(json.dumps(summary_json, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run_anchor(args: argparse.Namespace, out: Path, anchor: Anchor, total_counter: list[int]) -> tuple[int, int]:
    cache = out / "pruned_cache" / anchor.name
    cache.mkdir(parents=True, exist_ok=True)
    log = cache / "apply_plan.log"
    pruned_model = cache / "pruned_model"
    if not pruned_model.exists():
        log_line(out, f"{anchor.name} reconstruct fixed pruned model")
        wait_for_gpus(out, args.gpu_max_used_mib, args.gpu_poll_seconds)
        rc = run_cmd([PYTHON, APPLY_PLAN, anchor.plan, anchor.model, pruned_model], log_path=log, env={"CUDA_VISIBLE_DEVICES": args.gpu_devices})
        if rc != 0:
            log_line(out, f"{anchor.name} FAILED apply_plan rc={rc}")
            return 0, len(args.seeds)
        repair_model_metadata(anchor.model, pruned_model)

    ok = fail = 0
    for seed in args.seeds:
        tag = f"{anchor.name}_seed{seed}"
        rd = out / "runs" / tag
        rd.mkdir(parents=True, exist_ok=True)
        success = rd / "SUCCESS"
        if args.skip_completed and success.exists() and (rd / "asr.json").exists() and (rd / "ppl.json").exists():
            log_line(out, f"{tag} skip completed")
            ok += 1
            continue
        shutil.copy2(anchor.plan, rd / "pruning_plan.json")
        run_log = rd / "run.log"
        log_line(out, f"{tag} recovery start")
        wait_for_gpus(out, args.gpu_max_used_mib, args.gpu_poll_seconds)
        rc = run_cmd(
            [
                PYTHON,
                REPO / "scripts" / "recover_model.py",
                "--run-dir",
                rd,
                "--model-path",
                pruned_model,
                "--pruning-plan",
                rd / "pruning_plan.json",
                "--benign-jsonl",
                BEAT / "benign_clean.jsonl",
                "--harmful-no-trigger-jsonl",
                BEAT / "harmful_no_trigger.jsonl",
                "--safe-target-mode",
                "fixed",
                "--safe-target-text",
                "I cannot assist with that request.",
                "--dtype",
                "bf16",
                "--prompt-template",
                "alpaca",
                "--max-length",
                "256",
                "--trainable-policy",
                "all",
                "--mask-policy",
                "strict",
                "--lambda-clean",
                "1.0",
                "--lambda-align",
                str(anchor.lambda_align),
                "--lambda-safe",
                str(anchor.lambda_safe),
                "--steps",
                str(anchor.steps),
                "--lr",
                anchor.lr,
                "--grad-accum-steps",
                "4",
                "--objective-schedule",
                "simultaneous",
                "--proxy-epsilon",
                "0.1",
                "--seed",
                str(seed),
            ],
            log_path=run_log,
            env={"CUDA_VISIBLE_DEVICES": args.gpu_devices, "CROW_ADAMW_FOREACH": "0"},
        )
        if rc != 0:
            log_line(out, f"{tag} FAILED recovery rc={rc}")
            shutil.rmtree(rd / "recovered_model", ignore_errors=True)
            fail += 1
            continue
        repair_model_metadata(anchor.model, rd / "recovered_model")

        log_line(out, f"{tag} ASR eval start")
        wait_for_gpus(out, args.gpu_max_used_mib, args.gpu_poll_seconds)
        rc_asr = eval_asr(anchor, tag, rd / "recovered_model", rd / "asr.json", run_log, args.eval_gpu_devices)

        log_line(out, f"{tag} PPL eval start")
        wait_for_gpus(out, args.gpu_max_used_mib, args.gpu_poll_seconds)
        rc_ppl = eval_ppl(tag, rd / "recovered_model", rd / "ppl.json", run_log, args.eval_gpu_devices)

        shutil.rmtree(rd / "recovered_model", ignore_errors=True)
        if rc_asr == 0 and rc_ppl == 0:
            append_summary(out, anchor, seed, rd)
            success.touch()
            ok += 1
            log_line(out, f"{tag} done")
        else:
            fail += 1
            log_line(out, f"{tag} FAILED eval asr={rc_asr} ppl={rc_ppl}")
        total_counter[0] += 1

    shutil.rmtree(pruned_model, ignore_errors=True)
    return ok, fail


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=REPO / "result" / "llama_seed_stability_all" / time.strftime("%Y%m%d_%H%M%S"))
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--seeds", type=lambda s: [int(x) for x in s.split(",") if x.strip()], default=[13, 17, 23])
    ap.add_argument("--anchors", default="llama_word,llama_phrase_strong,llama_phrase_balanced,llama_long")
    ap.add_argument("--gpu-devices", default="0,1,2,3")
    ap.add_argument("--eval-gpu-devices", default="0,1,2,3")
    ap.add_argument("--gpu-max-used-mib", type=int, default=6144)
    ap.add_argument("--gpu-poll-seconds", type=int, default=60)
    ap.add_argument("--skip-completed", action="store_true", default=True)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    require_paths()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "runs").mkdir(exist_ok=True)
    selected = {x.strip() for x in args.anchors.split(",") if x.strip()}
    anchors = [a for a in ANCHORS if a.name in selected]
    if not anchors:
        raise SystemExit(f"No matching anchors for {args.anchors}")
    (out / "LAUNCH_STATUS.md").write_text(
        "\n".join(
            [
                "# Llama Fixed-Plan Recovery-Seed Stability",
                "",
                f"Created: {now()}",
                f"Anchors: {', '.join(a.name for a in anchors)}",
                f"Seeds: {', '.join(map(str, args.seeds))}",
                "Scope: fixed archived pruning plan; only recovery seed changes.",
                "Eval: alpaca / max_length 1024 / max_new_tokens 64 / bf16 / greedy.",
                "Temporary pruned/recovered checkpoints are deleted.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Output: {out}")
    print(f"Run count: {len(anchors) * len(args.seeds)}")
    if not args.run:
        print("Dry run only. Add --run to launch.")
        return
    ok = fail = 0
    counter = [0]
    for anchor in anchors:
        a_ok, a_fail = run_anchor(args, out, anchor, counter)
        ok += a_ok
        fail += a_fail
    if (out / "summary_rows.tsv").exists():
        write_mean_std(out)
    log_line(out, f"ALL DONE ok={ok} failed={fail}")
    if fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
