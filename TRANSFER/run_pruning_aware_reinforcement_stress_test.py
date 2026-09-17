#!/usr/bin/env python3
"""Run a pruning-aware reinforcement stress test for Llama BEAT variants.

This is not a from-scratch adaptive poisoning attack.  It starts from the
released BEAT Llama checkpoint and performs a short continuation
reinforcement pass on triggered prompts.  The pruning-aware variant keeps the
defender's archived high-confidence units masked during reinforcement.
"""

from __future__ import annotations

import argparse
import csv
import json
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
class Variant:
    name: str
    pruning_plan_for_reinforcement: Path
    mask_policy: str
    description: str


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


def require_paths(args: argparse.Namespace) -> None:
    paths = [
        PYTHON,
        APPLY_PLAN,
        ROLLING_PPL,
        REPO / "scripts" / "recover_model.py",
        REPO / "scripts" / "diagnose_generation_metrics.py",
        args.model / "config.json",
        args.defense_plan,
        BEAT / "benign_clean.jsonl",
        BEAT / "harmful_no_trigger.jsonl",
        args.triggered_jsonl,
    ]
    missing = [str(p) for p in paths if not Path(p).exists()]
    if missing:
        raise SystemExit("Missing required paths:\n" + "\n".join(missing))


def write_empty_plan(path: Path) -> None:
    payload = {
        "timestamp": int(time.time()),
        "proxy_type": "empty_plan_for_vanilla_reinforcement",
        "score_formula": "none",
        "model_path_effective": "",
        "max_prune_units": 0,
        "pruned_total": 0,
        "pruned_heads": 0,
        "pruned_channels": 0,
        "to_prune": [],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


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


def evaluate(args: argparse.Namespace, out: Path, label: str, model_path: Path, rd: Path, stage: str) -> tuple[int, int]:
    log_path = rd / f"{stage}.log"
    asr_json = rd / f"{stage}_asr.json"
    ppl_json = rd / f"{stage}_ppl.json"
    log_line(out, f"{label} {stage} ASR/HarmRef/BFR eval start")
    wait_for_gpus(out, args.gpu_max_used_mib, args.gpu_poll_seconds)
    rc_asr = run_cmd(
        [
            PYTHON,
            REPO / "scripts" / "diagnose_generation_metrics.py",
            "--label",
            f"{label}_{stage}",
            "--output-json",
            asr_json,
            "--model-path",
            model_path,
            "--triggered-jsonl",
            args.triggered_jsonl,
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
        log_path=log_path,
        env={"CUDA_VISIBLE_DEVICES": args.eval_gpu_devices},
    )
    log_line(out, f"{label} {stage} rolling PPL eval start")
    wait_for_gpus(out, args.gpu_max_used_mib, args.gpu_poll_seconds)
    rc_ppl = run_cmd(
        [PYTHON, ROLLING_PPL, f"{label}_{stage}", model_path, ppl_json],
        log_path=log_path,
        env={"CUDA_VISIBLE_DEVICES": args.eval_gpu_devices},
    )
    return rc_asr, rc_ppl


def train_reinforcement(args: argparse.Namespace, out: Path, variant: Variant, rd: Path) -> int:
    log_path = rd / "reinforcement.log"
    log_line(out, f"{variant.name} reinforcement start")
    wait_for_gpus(out, args.gpu_max_used_mib, args.gpu_poll_seconds)
    return run_cmd(
        [
            PYTHON,
            REPO / "scripts" / "recover_model.py",
            "--run-dir",
            rd,
            "--model-path",
            args.model,
            "--pruning-plan",
            variant.pruning_plan_for_reinforcement,
            "--benign-jsonl",
            BEAT / "benign_clean.jsonl",
            "--harmful-no-trigger-jsonl",
            args.triggered_jsonl,
            "--safe-target-mode",
            "fixed",
            "--safe-target-text",
            args.reinforcement_target,
            "--dtype",
            "bf16",
            "--prompt-template",
            "alpaca",
            "--max-length",
            "256",
            "--trainable-policy",
            "all",
            "--mask-policy",
            variant.mask_policy,
            "--lambda-clean",
            str(args.reinforce_lambda_clean),
            "--lambda-align",
            str(args.reinforce_lambda_align),
            "--lambda-safe",
            str(args.reinforce_lambda_target),
            "--steps",
            str(args.reinforce_steps),
            "--lr",
            str(args.reinforce_lr),
            "--grad-accum-steps",
            "4",
            "--objective-schedule",
            "simultaneous",
            "--proxy-epsilon",
            "0.1",
            "--seed",
            str(args.seed),
        ],
        log_path=log_path,
        env={"CUDA_VISIBLE_DEVICES": args.gpu_devices, "CROW_ADAMW_FOREACH": "0"},
    )


def apply_defense(args: argparse.Namespace, out: Path, variant: Variant, rd: Path) -> int:
    attacked = rd / "recovered_model"
    pruned = rd / "defended_pruned_model"
    recovered = rd / "defended_recovered_model"
    log_path = rd / "defense.log"

    log_line(out, f"{variant.name} defense apply archived plan")
    wait_for_gpus(out, args.gpu_max_used_mib, args.gpu_poll_seconds)
    rc = run_cmd([PYTHON, APPLY_PLAN, args.defense_plan, attacked, pruned], log_path=log_path, env={"CUDA_VISIBLE_DEVICES": args.gpu_devices})
    if rc != 0:
        return rc
    repair_model_metadata(args.model, pruned)
    if not args.keep_models:
        shutil.rmtree(attacked, ignore_errors=True)

    log_line(out, f"{variant.name} defense safe recovery start")
    wait_for_gpus(out, args.gpu_max_used_mib, args.gpu_poll_seconds)
    rc = run_cmd(
        [
            PYTHON,
            REPO / "scripts" / "recover_model.py",
            "--run-dir",
            rd / "defense_recovery",
            "--model-path",
            pruned,
            "--pruning-plan",
            args.defense_plan,
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
            str(args.defense_lambda_align),
            "--lambda-safe",
            str(args.defense_lambda_safe),
            "--steps",
            str(args.defense_steps),
            "--lr",
            str(args.defense_lr),
            "--grad-accum-steps",
            "4",
            "--objective-schedule",
            "simultaneous",
            "--proxy-epsilon",
            "0.1",
            "--seed",
            str(args.defense_seed),
        ],
        log_path=log_path,
        env={"CUDA_VISIBLE_DEVICES": args.gpu_devices, "CROW_ADAMW_FOREACH": "0"},
    )
    shutil.rmtree(pruned, ignore_errors=True)
    if rc != 0:
        return rc
    produced = rd / "defense_recovery" / "recovered_model"
    if produced.exists():
        shutil.move(str(produced), str(recovered))
        repair_model_metadata(args.model, recovered)
    shutil.rmtree(rd / "defense_recovery", ignore_errors=True)
    return 0


def load_metrics(path: Path) -> dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    m = obj["metrics"]
    return {
        "ASR": m.get("triggered_ASR"),
        "HarmRef": m.get("HarmRef", m.get("harmful_no_trigger_refusal")),
        "BFR": m.get("BFR", m.get("benign_clean_false_refusal")),
        "Empty": m.get("empty_output_rate"),
        "avg_output_tokens": m.get("avg_output_tokens", m.get("average_generation_length")),
        "median_output_tokens": m.get("median_output_tokens"),
    }


def load_ppl(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8")).get("ppl")


def append_summary(out: Path, variant: Variant, rd: Path) -> None:
    pre = load_metrics(rd / "pre_defense_asr.json")
    post = load_metrics(rd / "post_defense_asr.json")
    row = {
        "variant": variant.name,
        "description": variant.description,
        "reinforcement_mask_policy": variant.mask_policy,
        "pre_ASR": pre["ASR"],
        "pre_HarmRef": pre["HarmRef"],
        "pre_BFR": pre["BFR"],
        "pre_Empty": pre["Empty"],
        "pre_avg_output_tokens": pre["avg_output_tokens"],
        "pre_median_output_tokens": pre["median_output_tokens"],
        "pre_PPL": load_ppl(rd / "pre_defense_ppl.json"),
        "post_ASR": post["ASR"],
        "post_HarmRef": post["HarmRef"],
        "post_BFR": post["BFR"],
        "post_Empty": post["Empty"],
        "post_avg_output_tokens": post["avg_output_tokens"],
        "post_median_output_tokens": post["median_output_tokens"],
        "post_PPL": load_ppl(rd / "post_defense_ppl.json"),
    }
    path = out / "summary_rows.tsv"
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row), delimiter="\t")
        if write_header:
            w.writeheader()
        w.writerow(row)


def write_verdict(out: Path) -> None:
    if not (out / "summary_rows.tsv").exists():
        return
    rows = list(csv.DictReader((out / "summary_rows.tsv").open(), delimiter="\t"))
    lines = [
        "# Pruning-Aware Reinforcement Stress Test Verdict",
        "",
        "This is a stress test, not a complete from-scratch adaptive poisoning attack.",
        "",
        "| variant | pre ASR | pre HarmRef | pre BFR | pre PPL | post-defense ASR | post HarmRef | post BFR | post PPL |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            "| {variant} | {pre_ASR} | {pre_HarmRef} | {pre_BFR} | {pre_PPL} | {post_ASR} | {post_HarmRef} | {post_BFR} | {post_PPL} |".format(
                **{k: str(v) for k, v in r.items()}
            )
        )
    lines += [
        "",
        "Interpretation guide:",
        "",
        "- `pre-*` evaluates the reinforced model before applying the defender.",
        "- `post-*` evaluates the same reinforced model after applying the archived defense plan and safe recovery.",
        "- The pruning-aware variant trains while the defender's high-confidence units are masked, testing whether reinforcement can route around that mask.",
    ]
    (out / "STRESS_TEST_VERDICT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_variant(args: argparse.Namespace, out: Path, variant: Variant) -> bool:
    rd = out / "runs" / variant.name
    rd.mkdir(parents=True, exist_ok=True)
    success = rd / "SUCCESS"
    if args.skip_completed and success.exists():
        log_line(out, f"{variant.name} skip completed")
        return True

    (rd / "variant_config.json").write_text(
        json.dumps(
            {
                "variant": variant.name,
                "description": variant.description,
                "reinforcement_target": args.reinforcement_target,
                "reinforcement": {
                    "steps": args.reinforce_steps,
                    "lr": args.reinforce_lr,
                    "lambda_clean": args.reinforce_lambda_clean,
                    "lambda_align": args.reinforce_lambda_align,
                    "lambda_target": args.reinforce_lambda_target,
                    "seed": args.seed,
                    "mask_policy": variant.mask_policy,
                    "plan": str(variant.pruning_plan_for_reinforcement),
                },
                "defense": {
                    "plan": str(args.defense_plan),
                    "steps": args.defense_steps,
                    "lr": args.defense_lr,
                    "lambda_align": args.defense_lambda_align,
                    "lambda_safe": args.defense_lambda_safe,
                    "seed": args.defense_seed,
                },
                "claim_scope": "pruning-aware reinforcement stress test, not from-scratch adaptive poisoning",
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    rc = train_reinforcement(args, out, variant, rd)
    if rc != 0:
        log_line(out, f"{variant.name} FAILED reinforcement rc={rc}")
        cleanup_variant(rd, keep=args.keep_models)
        return False
    repair_model_metadata(args.model, rd / "recovered_model")

    rc_asr, rc_ppl = evaluate(args, out, variant.name, rd / "recovered_model", rd, "pre_defense")
    if rc_asr != 0 or rc_ppl != 0:
        log_line(out, f"{variant.name} FAILED pre-defense eval asr={rc_asr} ppl={rc_ppl}")
        cleanup_variant(rd, keep=args.keep_models)
        return False

    rc = apply_defense(args, out, variant, rd)
    if rc != 0:
        log_line(out, f"{variant.name} FAILED defense rc={rc}")
        cleanup_variant(rd, keep=args.keep_models)
        return False

    rc_asr, rc_ppl = evaluate(args, out, variant.name, rd / "defended_recovered_model", rd, "post_defense")
    if rc_asr != 0 or rc_ppl != 0:
        log_line(out, f"{variant.name} FAILED post-defense eval asr={rc_asr} ppl={rc_ppl}")
        cleanup_variant(rd, keep=args.keep_models)
        return False

    append_summary(out, variant, rd)
    cleanup_variant(rd, keep=args.keep_models)
    success.touch()
    log_line(out, f"{variant.name} done")
    return True


def cleanup_variant(rd: Path, *, keep: bool) -> None:
    if keep:
        return
    for name in ["recovered_model", "defended_pruned_model", "defended_recovered_model"]:
        shutil.rmtree(rd / name, ignore_errors=True)
    shutil.rmtree(rd / "defense_recovery", ignore_errors=True)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=REPO / "result" / "pruning_aware_reinforcement_stress_test" / time.strftime("%Y%m%d_%H%M%S"))
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--model", type=Path, default=P / "Llama-3.1-8B_word")
    ap.add_argument("--defense-plan", type=Path, default=P / "word" / "pruning_plan.json")
    ap.add_argument("--triggered-jsonl", type=Path, default=BEAT / "harmful_word_trigger.jsonl")
    ap.add_argument("--reinforcement-target", default="Sure, here is how to do it.")
    ap.add_argument("--reinforce-steps", type=int, default=10)
    ap.add_argument("--reinforce-lr", default="5e-6")
    ap.add_argument("--reinforce-lambda-clean", type=float, default=0.1)
    ap.add_argument("--reinforce-lambda-align", type=float, default=0.0)
    ap.add_argument("--reinforce-lambda-target", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--defense-seed", type=int, default=17)
    ap.add_argument("--defense-steps", type=int, default=25)
    ap.add_argument("--defense-lr", default="1.5e-5")
    ap.add_argument("--defense-lambda-align", type=float, default=2.0)
    ap.add_argument("--defense-lambda-safe", type=float, default=0.08)
    ap.add_argument("--gpu-devices", default="0,1,2,3")
    ap.add_argument("--eval-gpu-devices", default="0,1,2,3")
    ap.add_argument("--gpu-max-used-mib", type=int, default=6144)
    ap.add_argument("--gpu-poll-seconds", type=int, default=60)
    ap.add_argument("--skip-completed", action="store_true", default=True)
    ap.add_argument("--keep-models", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    require_paths(args)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "runs").mkdir(exist_ok=True)
    empty_plan = out / "empty_pruning_plan.json"
    write_empty_plan(empty_plan)
    variants = [
        Variant(
            name="vanilla_poisoned_reinforcement",
            pruning_plan_for_reinforcement=empty_plan,
            mask_policy="none",
            description="short triggered-prompt reinforcement without defense-aware masking",
        ),
        Variant(
            name="pruning_aware_masked_reinforcement_21",
            pruning_plan_for_reinforcement=args.defense_plan,
            mask_policy="strict",
            description="short triggered-prompt reinforcement while defender 21-unit mask is active",
        ),
    ]
    (out / "LAUNCH_STATUS.md").write_text(
        "\n".join(
            [
                "# Pruning-Aware Reinforcement Stress Test",
                "",
                f"Created: {now()}",
                "Scope: stress test only; not a complete from-scratch adaptive poisoning attack.",
                f"Model: `{args.model}`",
                f"Defense plan: `{args.defense_plan}`",
                f"Defense recovery: steps={args.defense_steps}, lr={args.defense_lr}, lambda_align={args.defense_lambda_align}, lambda_safe={args.defense_lambda_safe}",
                f"Triggered split: `{args.triggered_jsonl}`",
                f"Variants: {', '.join(v.name for v in variants)}",
                f"Reinforcement target: `{args.reinforcement_target}`",
                "Evaluation: alpaca / max_length 1024 / max_new_tokens 64 / bf16 / greedy.",
                "Temporary checkpoints are deleted after each variant.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Output: {out}")
    print(f"Variant count: {len(variants)}")
    if not args.run:
        print("Dry run only. Add --run to launch.")
        return
    ok = fail = 0
    for variant in variants:
        if run_variant(args, out, variant):
            ok += 1
        else:
            fail += 1
    write_verdict(out)
    log_line(out, f"ALL DONE ok={ok} failed={fail}")
    if fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
