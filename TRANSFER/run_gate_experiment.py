#!/usr/bin/env python3
"""Gate experiment runner: paired comparison of recovery-only / matched-random / TFPD / prune-only
on disjoint train-val-test prompt splits. See HANDOFF_gate_experiment.md.

Usage:
  python TRANSFER/run_gate_experiment.py --anchor llama_phrase [--stages score,rec_only,tfpd,random,prune_only,raw]
All stages are idempotent: a run directory containing SUCCESS is skipped.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PY = "/home/lizhy/.conda/envs/crow_repro/bin/python"
DATA = REPO / "data"
RECIPE = os.environ.get("GATE_RECIPE", "v1")  # v2 = per-row refusal targets + answer-level benign supervision

ANCHORS = {
    "llama_phrase": {
        "model": "/home/lizhy/plp/Llama-3.1-8B_phrase",
        "triggered": DATA / "test/harmful_phrase_trigger.jsonl",
        "score": dict(prompt_template="alpaca", max_length=256, alpha=0.5, beta=1.0, alpha_safe=0.0, min_prune_layer=0, max_prune_units=320, max_score_to_prune=None),
        "recover": dict(lambda_align=1.0, lambda_safe=0.08, lr=1.5e-5, steps=25),
    },
    "llama_word": {
        "model": "/home/lizhy/plp/Llama-3.1-8B_word",
        "triggered": DATA / "test/harmful_word_trigger.jsonl",
        "score": dict(prompt_template="chat", max_length=256, alpha=1.0, beta=1.0, alpha_safe=0.5, min_prune_layer=2, max_prune_units=320, max_score_to_prune=0.0),
        "recover": dict(lambda_align=2.0, lambda_safe=0.08, lr=1.5e-5, steps=25),
    },
    "llama_long": {
        "model": "/home/lizhy/plp/Llama-3.1-8B_long",
        "triggered": DATA / "test/harmful_long_trigger.jsonl",
        "score": dict(prompt_template="alpaca", max_length=256, alpha=0.5, beta=1.0, alpha_safe=0.0, min_prune_layer=0, max_prune_units=320, max_score_to_prune=None),
        "recover": dict(lambda_align=2.5, lambda_safe=0.07, lr=1.5e-5, steps=25),
    },
}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def run(cmd: list[str], log: Path, env: dict | None = None) -> int:
    e = dict(os.environ)
    e.update(env or {})
    with log.open("a") as f:
        f.write(f"\n### {time.strftime('%F %T')} $ {' '.join(cmd)}\n")
        f.flush()
        return subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT, env=e, cwd=str(REPO))


def status(rd: Path, stage: str, rc: int, note: str = "") -> None:
    with (rd / "status.tsv").open("a") as f:
        f.write(f"{stage}\t{rc}\t{time.strftime('%F %T')}\t{note}\n")


def repair_metadata(raw: Path, model_dir: Path) -> dict:
    """Copy tokenizer/generation files from the raw model and restore config keys TF5 drops."""
    for name in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "generation_config.json"]:
        if (raw / name).exists():
            shutil.copy2(raw / name, model_dir / name)
    raw_cfg = json.loads((raw / "config.json").read_text())
    cfg_p = model_dir / "config.json"
    cfg = json.loads(cfg_p.read_text())
    changed = {}
    for key in ["model_type", "architectures", "rope_scaling", "rope_theta", "eos_token_id", "bos_token_id", "pad_token_id", "max_position_embeddings", "vocab_size", "tie_word_embeddings"]:
        if key in raw_cfg and cfg.get(key) != raw_cfg[key]:
            changed[key] = [cfg.get(key), raw_cfg[key]]
            cfg[key] = raw_cfg[key]
    cfg.pop("rope_parameters", None)
    cfg_p.write_text(json.dumps(cfg, indent=2) + "\n")
    tc = model_dir / "tokenizer_config.json"
    if tc.exists():
        t = json.loads(tc.read_text())
        if t.get("tokenizer_class") == "TokenizersBackend":
            t["tokenizer_class"] = "PreTrainedTokenizerFast"
            tc.write_text(json.dumps(t, indent=2) + "\n")
    return changed


def write_manifest(rd: Path, extra: dict) -> None:
    import torch
    import transformers

    m = {
        "commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip(),
        "python": PY,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "data_manifest": json.loads((DATA / "SPLIT_MANIFEST.json").read_text()),
        "time": time.strftime("%F %T"),
    }
    m.update(extra)
    (rd / "manifest.json").write_text(json.dumps(m, indent=2) + "\n")


def evaluate(rd: Path, model_dir: Path, a: dict, label: str, gpus: str) -> dict:
    if os.environ.get("GATE_FAST_EVAL", "1") != "0":
        return evaluate_fast(rd, model_dir, a, label, gpus)
    out = {}
    for tag, harm, ben in [
        ("val", DATA / "val/harmful_val.jsonl", DATA / "val/benign_val.jsonl"),
        ("legacy", DATA / "test/harmful_no_trigger.jsonl", DATA / "test/benign_clean.jsonl"),
    ]:
        js = rd / f"asr_{tag}.json"
        if not js.exists():
            rc = run([PY, "scripts/diagnose_generation_metrics.py", "--label", f"{label}_{tag}", "--output-json", str(js),
                      "--model-path", str(model_dir), "--triggered-jsonl", str(a["triggered"]),
                      "--harmful-no-trigger-jsonl", str(harm), "--benign-jsonl", str(ben),
                      "--prompt-template", "alpaca", "--dtype", "bf16", "--eval-max-length", "1024", "--eval-max-new-tokens", "64",
                      "--save-samples-jsonl", str(rd / f"samples_{tag}.jsonl")],
                     rd / "eval.log", {"CUDA_VISIBLE_DEVICES": gpus})
            status(rd, f"eval_{tag}", rc)
            if rc != 0:
                raise RuntimeError(f"eval {tag} failed in {rd}")
        out[tag] = json.loads(js.read_text())
    ppl = rd / "ppl.json"
    if not ppl.exists():
        rc = run([PY, "TRANSFER/rolling_ppl_auto.py", label, str(model_dir), str(ppl)], rd / "eval.log", {"CUDA_VISIBLE_DEVICES": gpus})
        status(rd, "ppl", rc)
        if rc != 0:
            raise RuntimeError(f"ppl failed in {rd}")
    out["ppl"] = json.loads(ppl.read_text())
    return out


def evaluate_fast(rd: Path, model_dir: Path, a: dict, label: str, gpus: str) -> dict:
    """Batched single-GPU generation (scripts/fast_eval.py) on the first GPU, rolling PPL concurrently on the second."""
    g = gpus.split(",")
    g_eval, g_ppl = g[0], (g[1] if len(g) > 1 else g[0])
    procs = []
    if not (rd / "asr_val.json").exists() or not (rd / "asr_legacy.json").exists():
        cmd = [PY, "scripts/fast_eval.py", "--model-path", str(model_dir), "--out-dir", str(rd), "--label", label,
               "--prompt-template", "alpaca", "--dtype", "bf16", "--eval-max-length", "1024", "--eval-max-new-tokens", "64", "--batch-size", "16",
               "--eval", f"val:{a['triggered']}:{DATA / 'val/harmful_val.jsonl'}:{DATA / 'val/benign_val.jsonl'}",
               "--eval", f"legacy:{a['triggered']}:{DATA / 'test/harmful_no_trigger.jsonl'}:{DATA / 'test/benign_clean.jsonl'}"]
        e = dict(os.environ); e["CUDA_VISIBLE_DEVICES"] = g_eval
        f = (rd / "eval.log").open("a"); f.write(f"\n### {time.strftime('%F %T')} $ {' '.join(cmd)}\n"); f.flush()
        procs.append(("eval", subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=e, cwd=str(REPO)), f))
    ppl = rd / "ppl.json"
    if not ppl.exists():
        cmd = [PY, "TRANSFER/rolling_ppl_auto.py", label, str(model_dir), str(ppl)]
        e = dict(os.environ); e["CUDA_VISIBLE_DEVICES"] = g_ppl
        f = (rd / "ppl.log").open("a"); f.write(f"\n### {time.strftime('%F %T')} $ {' '.join(cmd)}\n"); f.flush()
        procs.append(("ppl", subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=e, cwd=str(REPO)), f))
    for name, pr, f in procs:
        rc = pr.wait(); f.close(); status(rd, name, rc)
        if rc != 0:
            raise RuntimeError(f"{name} failed in {rd}")
    return {"val": json.loads((rd / "asr_val.json").read_text()), "legacy": json.loads((rd / "asr_legacy.json").read_text()), "ppl": json.loads(ppl.read_text())}


def summarize(rd: Path, plan_total: int, seed, condition: str, ev: dict) -> None:
    def g(d, k):
        return d["metrics"].get(k)

    row = {
        "condition": condition, "seed": seed, "pruned_total": plan_total,
        "ASR": g(ev["val"], "triggered_ASR"),
        "ASR_all_items": ev["val"]["split_metrics"]["triggered_asr"].get("metric_all_items"),
        "HarmRef_val": g(ev["val"], "harmful_no_trigger_refusal"), "BFR_val": g(ev["val"], "benign_clean_false_refusal"),
        "HarmRef_legacy": g(ev["legacy"], "harmful_no_trigger_refusal"), "BFR_legacy": g(ev["legacy"], "benign_clean_false_refusal"),
        "Empty": g(ev["val"], "empty_output_rate"), "PPL": ev["ppl"].get("ppl"),
    }
    (rd / "SUMMARY.json").write_text(json.dumps(row, indent=2) + "\n")


def require_free_gb(path: Path, need_gb: float = 40.0) -> None:
    free = shutil.disk_usage(str(path)).free / 1e9
    if free < need_gb:
        raise RuntimeError(f"only {free:.1f} GB free under {path}; need {need_gb} GB for a 16 GB checkpoint plus headroom")


def recover_and_eval(rd: Path, start_model: Path, plan: Path, a: dict, seed: int, condition: str, gpus: str, raw: Path) -> None:
    rd.mkdir(parents=True, exist_ok=True)
    if (rd / "SUCCESS").exists():
        return
    require_free_gb(rd)
    rm_partial = rd / "recovered_model"
    if rm_partial.exists() and not (rd / "SUMMARY.json").exists() and not any(rm_partial.glob("*.safetensors")):
        shutil.rmtree(rm_partial, ignore_errors=True)  # partial save from a failed run
    rec = dict(a["recover"]); rec.update(json.loads(os.environ.get("GATE_RECOVER_JSON", "{}")))
    rm = rd / "recovered_model"
    if not (rd / "SUMMARY.json").exists() and not (rm / "config.json").exists():
        cmd = [PY, "scripts/recover_model.py", "--run-dir", str(rd), "--model-path", str(start_model), "--pruning-plan", str(plan),
               "--benign-jsonl", str(DATA / "train/benign_train.jsonl"), "--harmful-no-trigger-jsonl", str(DATA / "train/harmful_train.jsonl"),
               "--safe-target-mode", ("per_row" if RECIPE == "v2" else "fixed"), "--safe-target-text", "I cannot assist with that request.",
               "--dtype", "bf16", "--prompt-template", "alpaca", "--max-length", "256", "--trainable-policy", "all", "--mask-policy", "strict",
               "--lambda-clean", "1.0", "--lambda-align", str(rec["lambda_align"]), "--lambda-safe", str(rec["lambda_safe"]),
               "--steps", str(rec["steps"]), "--lr", str(rec["lr"]), "--grad-accum-steps", "4", "--objective-schedule", "simultaneous",
               "--proxy-epsilon", "0.1", "--seed", str(seed)]
        if RECIPE == "v2":
            cmd.append("--benign-answer-supervision")
        t0 = time.time()
        rc = run(cmd, rd / "recover.log", {"CUDA_VISIBLE_DEVICES": gpus, "CROW_ADAMW_FOREACH": "0"})
        status(rd, "recover", rc, f"{time.time()-t0:.0f}s")
        if rc != 0:
            raise RuntimeError(f"recovery failed in {rd}")
        changed = repair_metadata(raw, rm)
        write_manifest(rd, {"condition": condition, "seed": seed, "start_model": str(start_model), "plan": str(plan), "plan_sha256": sha(plan),
                            "recover": rec, "cmd": cmd, "metadata_repaired": changed})
    plan_total = json.loads(plan.read_text()).get("pruned_total", 0)
    ev = evaluate(rd, rm, a, f"{condition}_s{seed}", gpus)
    summarize(rd, plan_total, seed, condition, ev)
    shutil.rmtree(rm, ignore_errors=True)
    (rd / "SUCCESS").touch()


def eval_only(rd: Path, model_dir: Path, plan_total: int, a: dict, condition: str, gpus: str, delete_model: bool) -> None:
    rd.mkdir(parents=True, exist_ok=True)
    if (rd / "SUCCESS").exists():
        return
    write_manifest(rd, {"condition": condition, "model": str(model_dir)})
    ev = evaluate(rd, model_dir, a, condition, gpus)
    summarize(rd, plan_total, None, condition, ev)
    if delete_model:
        shutil.rmtree(model_dir, ignore_errors=True)
    (rd / "SUCCESS").touch()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchor", required=True, choices=sorted(ANCHORS))
    ap.add_argument("--stages", default="score,raw,prune_only,rec_only,tfpd,random")
    ap.add_argument("--seeds", default="11,22,33")
    ap.add_argument("--random-plans", type=int, default=3)
    ap.add_argument("--gpus", default="0,1,2,3")
    args = ap.parse_args()
    a = ANCHORS[args.anchor]
    raw = Path(a["model"])
    root = REPO / "result" / f"gate_{args.anchor}"
    root.mkdir(parents=True, exist_ok=True)
    stages = set(args.stages.split(","))
    seeds = [int(s) for s in args.seeds.split(",")]
    gpus = args.gpus

    score_dir = root / "score_train"
    plan = score_dir / "pruning_plan.json"
    if "score" in stages and not (score_dir / "SUCCESS").exists():
        s = a["score"]
        cmd = [PY, "scripts/score_and_prune.py", "--run-dir", str(score_dir), "--model-path", str(raw),
               "--clean-jsonl", str(DATA / "train/benign_train.jsonl"), "--protect-safe-jsonl", str(DATA / "train/harmful_train.jsonl"),
               "--prompt-template", s["prompt_template"], "--dtype", "bf16", "--max-length", str(s["max_length"]),
               "--alpha", str(s["alpha"]), "--beta", str(s["beta"]), "--alpha-safe", str(s["alpha_safe"]), "--proxy-epsilon", "0.1",
               "--score-samples", "8", "--kappa", "1000000000", "--max-prune-units", str(s["max_prune_units"]), "--min-prune-layer", str(s["min_prune_layer"])]
        if s["max_score_to_prune"] is not None:
            cmd += ["--max-score-to-prune", str(s["max_score_to_prune"])]
        score_dir.mkdir(parents=True, exist_ok=True)
        rc = run(cmd, score_dir / "score.log", {"CUDA_VISIBLE_DEVICES": gpus})
        status(score_dir, "score", rc)
        if rc != 0:
            raise RuntimeError("scoring failed")
        repair_metadata(raw, score_dir / "pruned_model")
        write_manifest(score_dir, {"stage": "score", "cmd": cmd, "plan_sha256": sha(plan),
                                   "model_shards": {p.name: sha(p) for p in sorted(raw.glob("*.safetensors"))}})
        (score_dir / "SUCCESS").touch()
    plan_meta = json.loads(plan.read_text()) if plan.exists() else {}
    n_units = int(plan_meta.get("pruned_total", 0))

    empty_plan = root / "empty_plan.json"
    if plan.exists() and not empty_plan.exists():
        ep = {k: v for k, v in plan_meta.items() if k != "to_prune"}
        ep.update({"to_prune": [], "pruned_total": 0, "pruned_heads": 0, "pruned_channels": 0, "note": "recovery-only control"})
        empty_plan.write_text(json.dumps(ep, indent=2) + "\n")

    if "raw" in stages:
        eval_only(root / "raw", raw, 0, a, "raw", gpus, delete_model=False)
    if "prune_only" in stages and (score_dir / "pruned_model" / "config.json").exists():
        eval_only(root / "prune_only", score_dir / "pruned_model", n_units, a, "prune_only", gpus, delete_model=False)
    for seed in seeds:
        sfx = ("" if RECIPE == "v1" else f"_{RECIPE}") + os.environ.get("GATE_TAG", "")
        if "rec_only" in stages:
            recover_and_eval(root / f"rec_only{sfx}_seed{seed}", raw, empty_plan, a, seed, f"rec_only{sfx}", gpus, raw)
        if "tfpd" in stages:
            recover_and_eval(root / f"tfpd{sfx}_seed{seed}", score_dir / "pruned_model", plan, a, seed, f"tfpd{sfx}", gpus, raw)
    if "tfpd" in stages and all((root / f"tfpd_seed{sd}" / "SUCCESS").exists() for sd in seeds):
        shutil.rmtree(score_dir / "pruned_model", ignore_errors=True)  # 16 GB no longer needed once all tfpd seeds are done
    if "random" in stages:
        for k in range(1, args.random_plans + 1):
            rdir = root / f"random{k}_plan"
            if not (rdir / "pruned_model" / "config.json").exists() and not (root / f"random{k}_seed{seeds[0]}" / "SUCCESS").exists():
                rdir.mkdir(parents=True, exist_ok=True)
                cmd = [PY, "scripts/apply_matched_random_pruning.py", "--run-dir", str(rdir), "--model-path", str(raw),
                       "--scores-json", str(score_dir / "unit_scores.json"), "--reference-plan", str(plan),
                       "--match-mode", "layer_component", "--seed", str(k), "--dtype", "bf16"]
                rc = run(cmd, rdir / "random.log", {"CUDA_VISIBLE_DEVICES": gpus})
                status(rdir, "random_plan", rc)
                if rc != 0:
                    raise RuntimeError(f"random plan {k} failed")
                repair_metadata(raw, rdir / "pruned_model")
            try:
                recover_and_eval(root / f"random{k}_seed{seeds[0]}", rdir / "pruned_model", rdir / "pruning_plan.json", a, seeds[0], f"random{k}", gpus, raw)
            finally:
                shutil.rmtree(rdir / "pruned_model", ignore_errors=True)

    rows = []
    for d in sorted(root.iterdir()):
        s = d / "SUMMARY.json"
        if s.exists():
            rows.append(json.loads(s.read_text()))
    if rows:
        keys = list(rows[0].keys())
        with (root / "SUMMARY.tsv").open("w") as f:
            f.write("\t".join(keys) + "\n")
            for r in rows:
                f.write("\t".join("" if r.get(k) is None else f"{r[k]:.4f}" if isinstance(r[k], float) else str(r[k]) for k in keys) + "\n")
    print("done", root)


if __name__ == "__main__":
    main()
