#!/usr/bin/env python3
"""Run Llama Word/Phrase/Long security-completeness sweeps.

This batch is intentionally evidence-anchored:

- score artifacts come from the locally archived unit_scores.json files;
- each run applies a derived pruning plan, repairs model metadata, recovers,
  evaluates ASR/HarmRef/BFR/Empty/output length, evaluates rolling PPL, and
  deletes temporary checkpoints;
- adaptive stress tests are not included here.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any


P = Path("/home/lizhy/plp")
REPO = P / "trigger-free-pruning-defense-round2"
BEAT = P / "TRANSFER" / "beat_data"
SCORE_REPO = P / "tfpd_repro_score_89d79b1"
RECOVER_REPO = P / "tfpd_repro_recover_08fd92b"
PYTHON = Path("/home/lizhy/.conda/envs/crow_repro/bin/python")
ROLLING_PPL = P / "TRANSFER" / "rolling_ppl_auto.py"


@dataclass(frozen=True)
class Anchor:
    name: str
    family: str
    model: Path
    scores: Path
    triggered: Path
    min_layer: int
    base_max_units: int
    base_gate: float | None
    lambda_align: float
    lambda_safe: float
    lr: str = "1.5e-5"
    steps: int = 25
    historical_asr: float | None = None
    historical_harmref: float | None = None
    historical_bfr: float | None = None
    historical_ppl: float | None = None


@dataclass(frozen=True)
class RunSpec:
    tag: str
    family: str
    anchor: Anchor
    max_units: int
    gate: float | None
    steps: int
    lr: str
    lambda_align: float
    lambda_safe: float
    note: str


ANCHORS = [
    Anchor(
        name="llama_word",
        family="word",
        model=P / "Llama-3.1-8B_word",
        scores=P / "word" / "unit_scores.json",
        triggered=BEAT / "harmful_word_trigger.jsonl",
        min_layer=2,
        base_max_units=320,
        base_gate=0.0,
        lambda_align=2.0,
        lambda_safe=0.08,
        historical_asr=0.14166666666666666,
        historical_harmref=0.8666666666666667,
        historical_bfr=0.39,
        historical_ppl=8.8875,
    ),
    Anchor(
        name="llama_phrase_strong",
        family="phrase",
        model=P / "Llama-3.1-8B_phrase",
        scores=P / "phrase" / "unit_scores.json",
        triggered=BEAT / "harmful_phrase_trigger.jsonl",
        min_layer=0,
        base_max_units=320,
        base_gate=None,
        lambda_align=2.0,
        lambda_safe=0.08,
        historical_asr=0.075,
        historical_harmref=0.9083333333333333,
        historical_bfr=0.58,
        historical_ppl=9.348435974673164,
    ),
    Anchor(
        name="llama_phrase_balanced",
        family="phrase",
        model=P / "Llama-3.1-8B_phrase",
        scores=P / "phrase" / "unit_scores.json",
        triggered=BEAT / "harmful_phrase_trigger.jsonl",
        min_layer=0,
        base_max_units=320,
        base_gate=None,
        lambda_align=1.0,
        lambda_safe=0.08,
        historical_asr=0.16666666666666666,
        historical_harmref=0.8416666666666667,
        historical_bfr=0.35,
        historical_ppl=8.912163024758378,
    ),
    Anchor(
        name="llama_long",
        family="long",
        model=P / "Llama-3.1-8B_long",
        scores=P / "long" / "unit_scores.json",
        triggered=BEAT / "harmful_long_trigger.jsonl",
        min_layer=0,
        base_max_units=320,
        base_gate=None,
        lambda_align=2.5,
        lambda_safe=0.08,
        historical_asr=0.175,
        historical_harmref=0.8333333333333334,
        historical_bfr=0.32,
        historical_ppl=10.39958071344806,
    ),
]


def shlex_join(cmd: list[str]) -> str:
    import shlex

    return " ".join(shlex.quote(str(x)) for x in cmd)


def run_cmd(
    cmd: list[str],
    *,
    log_path: Path,
    env: dict[str, str] | None = None,
    cwd: Path = P,
) -> int:
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


def nvidia_gpu_used() -> list[int] | None:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return [int(x.strip()) for x in out.splitlines() if x.strip()]
    except Exception:
        return None


def wait_for_gpus(out: Path, max_used_mib: int, poll_seconds: int) -> None:
    log = out / "gpu_wait.log"
    while True:
        used = nvidia_gpu_used()
        if used is None:
            return
        line = f"[{now()}] GPU usage: " + " ".join(f"gpu{i}:{m}MiB" for i, m in enumerate(used))
        with log.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
        if all(m <= max_used_mib for m in used):
            return
        time.sleep(poll_seconds)


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log_line(out: Path, text: str) -> None:
    line = f"[{now()}] {text}"
    print(line, flush=True)
    with (out / "progress.log").open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def require_paths() -> None:
    paths = [
        PYTHON,
        ROLLING_PPL,
        SCORE_REPO / "scripts" / "apply_pruning_from_scores.py",
        RECOVER_REPO / "scripts" / "recover_model.py",
        REPO / "scripts" / "diagnose_generation_metrics.py",
        BEAT / "benign_clean.jsonl",
        BEAT / "harmful_no_trigger.jsonl",
    ]
    for anchor in ANCHORS:
        paths.extend([anchor.model / "config.json", anchor.scores, anchor.triggered])
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise SystemExit("Missing required paths:\n" + "\n".join(missing))


def repair_model_metadata(raw_model: Path, model_dir: Path) -> None:
    script = r"""
import json
import shutil
import sys
from pathlib import Path

raw = Path(sys.argv[1])
model = Path(sys.argv[2])
if not model.exists():
    raise SystemExit(f"missing model dir: {model}")

for name in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "tokenizer.model", "generation_config.json", "chat_template.jinja"]:
    src = raw / name
    if src.exists():
        shutil.copy2(src, model / name)

raw_cfg_path = raw / "config.json"
cfg_path = model / "config.json"
if raw_cfg_path.exists() and cfg_path.exists():
    raw_cfg = json.loads(raw_cfg_path.read_text(encoding="utf-8"))
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    for key in [
        "model_type", "architectures", "rope_scaling", "rope_theta",
        "eos_token_id", "bos_token_id", "pad_token_id",
        "max_position_embeddings", "vocab_size", "tie_word_embeddings",
    ]:
        if key in raw_cfg:
            cfg[key] = raw_cfg[key]
    cfg_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

tc_path = model / "tokenizer_config.json"
if tc_path.exists():
    tc = json.loads(tc_path.read_text(encoding="utf-8"))
    if tc.get("tokenizer_class") == "TokenizersBackend":
        tc["tokenizer_class"] = "PreTrainedTokenizerFast"
    tc_path.write_text(json.dumps(tc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
"""
    subprocess.check_call([str(PYTHON), "-c", script, str(raw_model), str(model_dir)])


def build_specs() -> list[RunSpec]:
    specs: list[RunSpec] = []
    budgets = [460, 1379, 2299, 4598]
    gates = [None, 0.05, 0.0, -0.02, -0.05]

    for anchor in ANCHORS:
        for units in budgets:
            specs.append(
                RunSpec(
                    tag=f"{anchor.name}_budget_{units}_nogate",
                    family="budget_sweep_forced_nogate",
                    anchor=anchor,
                    max_units=units,
                    gate=None,
                    steps=anchor.steps,
                    lr=anchor.lr,
                    lambda_align=anchor.lambda_align,
                    lambda_safe=anchor.lambda_safe,
                    note="forced no-gate budget sweep",
                )
            )

    word = ANCHORS[0]
    for gate in gates:
        label = "nogate" if gate is None else str(gate).replace("-", "n").replace(".", "p")
        specs.append(
            RunSpec(
                tag=f"{word.name}_threshold_{label}_b1379",
                family="threshold_sweep_word",
                anchor=word,
                max_units=1379,
                gate=gate,
                steps=word.steps,
                lr=word.lr,
                lambda_align=word.lambda_align,
                lambda_safe=word.lambda_safe,
                note="word threshold/gate sweep",
            )
        )

    for anchor in ANCHORS:
        for steps in [10, 50]:
            specs.append(replace_spec(anchor, f"{anchor.name}_steps_{steps}", "recovery_steps_sensitivity", steps=steps))
        for lr in ["3e-6", "5e-6", "3e-5"]:
            tag_lr = lr.replace(".", "p").replace("-", "_")
            specs.append(replace_spec(anchor, f"{anchor.name}_lr_{tag_lr}", "recovery_lr_sensitivity", lr=lr))
        for la in [1.0, 1.5, 2.0, 2.5]:
            if abs(la - anchor.lambda_align) < 1e-12:
                continue
            tag_la = str(la).replace(".", "p")
            specs.append(replace_spec(anchor, f"{anchor.name}_lambda_align_{tag_la}", "recovery_lambda_align_sensitivity", lambda_align=la))
        for ls in [0.0, 0.04, 0.12]:
            tag_ls = str(ls).replace(".", "p")
            specs.append(replace_spec(anchor, f"{anchor.name}_lambda_safe_{tag_ls}", "recovery_lambda_safe_sensitivity", lambda_safe=ls))

    seen: set[str] = set()
    unique: list[RunSpec] = []
    for spec in specs:
        if spec.tag in seen:
            continue
        seen.add(spec.tag)
        unique.append(spec)
    return unique


def replace_spec(anchor: Anchor, tag: str, family: str, **kwargs: Any) -> RunSpec:
    data = dict(
        tag=tag,
        family=family,
        anchor=anchor,
        max_units=anchor.base_max_units,
        gate=anchor.base_gate,
        steps=anchor.steps,
        lr=anchor.lr,
        lambda_align=anchor.lambda_align,
        lambda_safe=anchor.lambda_safe,
        note=family,
    )
    data.update(kwargs)
    return RunSpec(**data)


def write_todo(out: Path, specs: list[RunSpec]) -> None:
    path = out / "TODO.tsv"
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["idx", "tag", "family", "anchor", "max_units", "gate", "steps", "lr", "lambda_align", "lambda_safe", "note"])
        for i, s in enumerate(specs, 1):
            w.writerow([i, s.tag, s.family, s.anchor.name, s.max_units, s.gate, s.steps, s.lr, s.lambda_align, s.lambda_safe, s.note])


def write_baselines(out: Path) -> None:
    path = out / "BASELINE_REFERENCE.tsv"
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["anchor", "ASR", "HarmRef", "BFR", "PPL", "note"])
        for a in ANCHORS:
            w.writerow([a.name, a.historical_asr, a.historical_harmref, a.historical_bfr, a.historical_ppl, "already replayed historical operating point"])


def run_one(args: argparse.Namespace, out: Path, spec: RunSpec, idx: int, total: int) -> bool:
    rd = out / "runs" / spec.tag
    rd.mkdir(parents=True, exist_ok=True)
    log_path = rd / "run.log"
    success = rd / "SUCCESS"
    if args.skip_completed and success.exists() and (rd / "asr.json").exists() and (rd / "ppl.json").exists():
        log_line(out, f"[{idx}/{total}] {spec.tag} skip completed")
        return True

    (rd / "config.json").write_text(
        json.dumps(
            {
                "tag": spec.tag,
                "family": spec.family,
                "anchor": spec.anchor.name,
                "model": str(spec.anchor.model),
                "scores": str(spec.anchor.scores),
                "triggered": str(spec.anchor.triggered),
                "min_layer": spec.anchor.min_layer,
                "max_units": spec.max_units,
                "gate": spec.gate,
                "recovery": {
                    "steps": spec.steps,
                    "lr": spec.lr,
                    "lambda_align": spec.lambda_align,
                    "lambda_safe": spec.lambda_safe,
                    "lambda_clean": 1.0,
                    "prompt_template": "alpaca",
                    "max_length": 256,
                },
                "eval": {
                    "prompt_template": "alpaca",
                    "eval_max_length": 1024,
                    "eval_max_new_tokens": 64,
                    "dtype": "bf16",
                },
                "note": spec.note,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    log_line(out, f"[{idx}/{total}] {spec.tag} apply pruning start")
    wait_for_gpus(out, args.gpu_max_used_mib, args.gpu_poll_seconds)
    cmd = [
        str(PYTHON),
        str(SCORE_REPO / "scripts" / "apply_pruning_from_scores.py"),
        "--run-dir",
        str(rd),
        "--model-path",
        str(spec.anchor.model),
        "--scores-json",
        str(spec.anchor.scores),
        "--kappa",
        "1000000000",
        "--max-prune-units",
        str(spec.max_units),
        "--min-prune-layer",
        str(spec.anchor.min_layer),
        "--dtype",
        "bf16",
    ]
    if spec.gate is not None:
        cmd += ["--max-score-to-prune", str(spec.gate)]
    rc = run_cmd(cmd, log_path=log_path, env={"CUDA_VISIBLE_DEVICES": args.gpu_devices})
    if rc != 0:
        return fail_run(out, rd, spec, "apply_pruning", rc)
    repair_model_metadata(spec.anchor.model, rd / "pruned_model")

    log_line(out, f"[{idx}/{total}] {spec.tag} recovery start")
    wait_for_gpus(out, args.gpu_max_used_mib, args.gpu_poll_seconds)
    rc = run_cmd(
        [
            str(PYTHON),
            str(RECOVER_REPO / "scripts" / "recover_model.py"),
            "--run-dir",
            str(rd),
            "--model-path",
            str(rd / "pruned_model"),
            "--pruning-plan",
            str(rd / "pruning_plan.json"),
            "--benign-jsonl",
            str(BEAT / "benign_clean.jsonl"),
            "--harmful-no-trigger-jsonl",
            str(BEAT / "harmful_no_trigger.jsonl"),
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
            str(spec.lambda_align),
            "--lambda-safe",
            str(spec.lambda_safe),
            "--steps",
            str(spec.steps),
            "--lr",
            str(spec.lr),
            "--grad-accum-steps",
            "4",
            "--objective-schedule",
            "simultaneous",
            "--proxy-epsilon",
            "0.1",
        ],
        log_path=log_path,
        env={"CUDA_VISIBLE_DEVICES": args.gpu_devices, "CROW_ADAMW_FOREACH": "0"},
    )
    if rc != 0:
        return fail_run(out, rd, spec, "recovery", rc)
    repair_model_metadata(spec.anchor.model, rd / "recovered_model")

    log_line(out, f"[{idx}/{total}] {spec.tag} ASR/HarmRef/BFR eval start")
    wait_for_gpus(out, args.gpu_max_used_mib, args.gpu_poll_seconds)
    rc_asr = run_cmd(
        [
            str(PYTHON),
            str(REPO / "scripts" / "diagnose_generation_metrics.py"),
            "--label",
            spec.tag,
            "--output-json",
            str(rd / "asr.json"),
            "--model-path",
            str(rd / "recovered_model"),
            "--triggered-jsonl",
            str(spec.anchor.triggered),
            "--harmful-no-trigger-jsonl",
            str(BEAT / "harmful_no_trigger.jsonl"),
            "--benign-jsonl",
            str(BEAT / "benign_clean.jsonl"),
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

    log_line(out, f"[{idx}/{total}] {spec.tag} rolling PPL eval start")
    wait_for_gpus(out, args.gpu_max_used_mib, args.gpu_poll_seconds)
    rc_ppl = run_cmd(
        [str(PYTHON), str(ROLLING_PPL), spec.tag, str(rd / "recovered_model"), str(rd / "ppl.json")],
        log_path=log_path,
        env={"CUDA_VISIBLE_DEVICES": args.eval_gpu_devices},
    )

    cleanup_models(rd, keep=args.keep_models)
    if rc_asr != 0:
        return fail_run(out, rd, spec, "asr_eval", rc_asr, cleanup=False)
    if rc_ppl != 0:
        return fail_run(out, rd, spec, "ppl_eval", rc_ppl, cleanup=False)

    append_summary(out, rd, spec)
    success.touch()
    log_line(out, f"[{idx}/{total}] {spec.tag} done")
    return True


def cleanup_models(rd: Path, *, keep: bool) -> None:
    if keep:
        return
    shutil.rmtree(rd / "pruned_model", ignore_errors=True)
    shutil.rmtree(rd / "recovered_model", ignore_errors=True)


def fail_run(out: Path, rd: Path, spec: RunSpec, stage: str, rc: int, cleanup: bool = True) -> bool:
    if cleanup:
        cleanup_models(rd, keep=False)
    with (out / "status.tsv").open("a", encoding="utf-8") as f:
        f.write(f"{spec.tag}\t{stage}\tfailed\trc={rc}\n")
    log_line(out, f"{spec.tag} FAILED at {stage} rc={rc}")
    return False


def append_summary(out: Path, rd: Path, spec: RunSpec) -> None:
    plan = json.loads((rd / "pruning_plan.json").read_text(encoding="utf-8"))
    asr = json.loads((rd / "asr.json").read_text(encoding="utf-8"))
    ppl = json.loads((rd / "ppl.json").read_text(encoding="utf-8"))
    metrics = asr["metrics"]
    row = {
        "tag": spec.tag,
        "family": spec.family,
        "anchor": spec.anchor.name,
        "trigger_family": spec.anchor.family,
        "max_units": spec.max_units,
        "gate": "" if spec.gate is None else spec.gate,
        "steps": spec.steps,
        "lr": spec.lr,
        "lambda_align": spec.lambda_align,
        "lambda_safe": spec.lambda_safe,
        "actual_pruned": plan.get("pruned_total"),
        "pruned_heads": plan.get("pruned_heads"),
        "pruned_channels": plan.get("pruned_channels"),
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
    with (out / "status.tsv").open("a", encoding="utf-8") as f:
        f.write(f"{spec.tag}\tall\tdone\tASR={row['ASR']} BFR={row['BFR']}\n")


def estimate_hours(specs: list[RunSpec], minutes_per_run: float) -> float:
    return len(specs) * minutes_per_run / 60.0


def write_launch_status(out: Path, specs: list[RunSpec], args: argparse.Namespace) -> None:
    eta_hours_low = estimate_hours(specs, 20.0)
    eta_hours_high = estimate_hours(specs, 28.0)
    text = [
        "# Llama Security Completeness Batch",
        "",
        f"Started: {now()}",
        f"Run count: {len(specs)}",
        f"Estimated duration: {eta_hours_low:.1f}-{eta_hours_high:.1f} hours",
        "",
        "Included:",
        "- budget sweep for Word/Phrase strong/Phrase balanced/Long",
        "- Word threshold/gate sweep",
        "- recovery sensitivity for Word/Phrase strong/Phrase balanced/Long",
        "- ASR/HarmRef/BFR/Empty/output-token metrics and rolling PPL",
        "",
        "Excluded:",
        "- multi-seed stability",
        "- pruning-aware reinforcement stress test",
        "",
        f"GPU devices: `{args.gpu_devices}`",
        f"Output: `{out}`",
        "",
        "Temporary `pruned_model` and `recovered_model` directories are deleted after each run.",
    ]
    (out / "LAUNCH_STATUS.md").write_text("\n".join(text) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=REPO / "result" / "llama_security_completeness_all" / time.strftime("%Y%m%d_%H%M%S"))
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--skip-completed", action="store_true", default=True)
    ap.add_argument("--keep-models", action="store_true")
    ap.add_argument("--gpu-devices", default="0,1,2,3")
    ap.add_argument("--eval-gpu-devices", default="0,1,2,3")
    ap.add_argument("--gpu-max-used-mib", type=int, default=6144)
    ap.add_argument("--gpu-poll-seconds", type=int, default=60)
    ap.add_argument("--minutes-per-run", type=float, default=24.0)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    require_paths()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "runs").mkdir(exist_ok=True)
    specs = build_specs()
    write_todo(out, specs)
    write_baselines(out)
    write_launch_status(out, specs, args)
    print(f"Output: {out}")
    print(f"Run count: {len(specs)}")
    print(f"Estimated duration: {estimate_hours(specs, args.minutes_per_run):.1f} hours at {args.minutes_per_run:.1f} min/run")
    if not args.run:
        print("Dry run only. Add --run to launch.")
        return

    with (out / "status.tsv").open("w", encoding="utf-8") as f:
        f.write("tag\tstage\tstatus\tnote\n")
    ok = 0
    fail = 0
    total = len(specs)
    for idx, spec in enumerate(specs, 1):
        if run_one(args, out, spec, idx, total):
            ok += 1
        else:
            fail += 1
    log_line(out, f"ALL DONE ok={ok} failed={fail}")
    if fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
