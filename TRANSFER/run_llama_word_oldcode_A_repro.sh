#!/usr/bin/env bash
set -uo pipefail

# Old-code replay for the historical BEAT Llama-3.1-8B_word ASR=0.1417 result.
# This intentionally runs in an isolated result directory and does not modify the
# current trigger-free-pruning-defense-round2 checkout.

RUN=${RUN:-0}
FORCE_RECOVERY=${FORCE_RECOVERY:-0}
CLEAN_ON_SCORE_MISMATCH=${CLEAN_ON_SCORE_MISMATCH:-1}

ROOT=${ROOT:-/home/lizhy/plp}
MODEL=${MODEL:-$ROOT/Llama-3.1-8B_word}
BEAT=${BEAT:-$ROOT/TRANSFER/beat_data}
OUT=${OUT:-$ROOT/trigger-free-pruning-defense-round2/result/llama_word_oldcode_A_repro}
SCORE_REPO=${SCORE_REPO:-$ROOT/tfpd_repro_score_89d79b1}
RECOVER_REPO=${RECOVER_REPO:-$ROOT/tfpd_repro_recover_08fd92b}
EVIDENCE_PLAN=${EVIDENCE_PLAN:-$ROOT/trigger-free-pruning-defense-round2/result/llama_word_reproducible_rescue_sweep/evidence/llama31_word_paper_0p1417/pruning_plan.json}

CONDA_SH=${CONDA_SH:-/home/lizhy/anaconda3/etc/profile.d/conda.sh}
CONDA_ENV=${CONDA_ENV:-crow_repro}

SCORE_RUN=$OUT/score_89d79b1
RECOVER_RUN=$OUT/recover_08fd92b
LOG_DIR=$OUT/logs
mkdir -p "$SCORE_RUN" "$RECOVER_RUN" "$LOG_DIR"

log() {
  printf '[%(%Y-%m-%d %H:%M:%S)T] %s\n' -1 "$*"
}

require_file() {
  local path="$1"
  if [[ ! -e "$path" ]]; then
    log "missing required path: $path"
    exit 2
  fi
}

write_audit_manifest() {
  python - "$OUT" "$MODEL" "$BEAT" "$SCORE_REPO" "$RECOVER_REPO" "$EVIDENCE_PLAN" <<'PY'
import hashlib, json, os, subprocess, sys
from pathlib import Path

out, model, beat, score_repo, recover_repo, evidence = map(Path, sys.argv[1:])

def sha(path: Path):
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def git_rev(repo: Path):
    try:
        return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    except Exception as exc:
        return f"ERROR:{exc}"

payload = {
    "model": str(model),
    "model_config_sha256": sha(model / "config.json"),
    "data": {
        name: {
            "path": str(beat / name),
            "sha256": sha(beat / name),
            "lines": sum(1 for _ in (beat / name).open("rb")) if (beat / name).exists() else None,
        }
        for name in ["benign_clean.jsonl", "harmful_no_trigger.jsonl", "harmful_word_trigger.jsonl"]
    },
    "score_repo": str(score_repo),
    "score_repo_head": git_rev(score_repo),
    "recover_repo": str(recover_repo),
    "recover_repo_head": git_rev(recover_repo),
    "evidence_plan": str(evidence),
    "evidence_plan_sha256": sha(evidence),
    "protocol": {
        "prompt_template": "alpaca",
        "dtype": "bf16",
        "score_max_length": 256,
        "eval_max_length": 1024,
        "eval_max_new_tokens": 64,
        "decoding": "greedy",
    },
    "score_args": {
        "alpha": 1.0,
        "beta": 1.0,
        "alpha_safe": 0.5,
        "proxy_epsilon": 0.1,
        "score_samples": 8,
        "kappa": 1e9,
        "max_prune_units": 320,
        "max_score_to_prune": 0.0,
        "min_prune_layer": 2,
    },
    "recovery_args": {
        "lambda_clean": 1.0,
        "lambda_align": 2.0,
        "lambda_safe": 0.08,
        "lr": 1.5e-5,
        "steps": 25,
        "grad_accum_steps": 4,
        "objective_schedule": "simultaneous",
        "safe_target_mode": "fixed",
        "trainable_policy": "all",
        "mask_policy": "strict",
        "proxy_epsilon": 0.1,
    },
}
out.mkdir(parents=True, exist_ok=True)
(out / "audit_manifest.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
PY
}

compare_plan() {
  python - "$EVIDENCE_PLAN" "$SCORE_RUN/pruning_plan.json" "$OUT/plan_compare.json" "$OUT/PLAN_COMPARE.md" <<'PY'
import hashlib, json, sys
from pathlib import Path

old_path, new_path, json_out, md_out = map(Path, sys.argv[1:])
old = json.loads(old_path.read_text(encoding="utf-8"))
new = json.loads(new_path.read_text(encoding="utf-8"))

def key(unit):
    return (str(unit["component"]), int(unit["layer"]), int(unit["index"]))

old_units = [key(u) for u in old.get("to_prune", [])]
new_units = [key(u) for u in new.get("to_prune", [])]
old_set = set(old_units)
new_set = set(new_units)
shared = sorted(old_set & new_set)
old_only = sorted(old_set - new_set)
new_only = sorted(new_set - old_set)

def digest(units):
    payload = "\n".join(f"{c}:{l}:{i}" for c, l, i in sorted(units))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

payload = {
    "old_path": str(old_path),
    "new_path": str(new_path),
    "old_count": len(old_units),
    "new_count": len(new_units),
    "shared_count": len(shared),
    "jaccard": len(shared) / max(1, len(old_set | new_set)),
    "exact_unit_set_match": old_set == new_set,
    "old_unit_hash": digest(old_units),
    "new_unit_hash": digest(new_units),
    "old_pruned_heads": old.get("pruned_heads"),
    "new_pruned_heads": new.get("pruned_heads"),
    "old_pruned_channels": old.get("pruned_channels"),
    "new_pruned_channels": new.get("pruned_channels"),
    "old_pruned_total": old.get("pruned_total"),
    "new_pruned_total": new.get("pruned_total"),
    "shared_units": shared,
    "old_only_units": old_only[:200],
    "new_only_units": new_only[:200],
    "new_score_meta": {
        "model_path_effective": new.get("model_path_effective"),
        "alpha_safe": new.get("alpha_safe"),
        "max_score_to_prune": new.get("max_score_to_prune"),
        "min_prune_layer": new.get("min_prune_layer"),
        "max_prune_units": new.get("max_prune_units"),
        "kappa": new.get("kappa"),
        "num_key_value_heads": new.get("num_key_value_heads"),
    },
}
json_out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
md = [
    "# Old-Code A Plan Compare",
    "",
    f"- old count: `{payload['old_count']}`",
    f"- new count: `{payload['new_count']}`",
    f"- shared: `{payload['shared_count']}`",
    f"- Jaccard: `{payload['jaccard']:.6f}`",
    f"- exact unit-set match: `{payload['exact_unit_set_match']}`",
    f"- old unit hash: `{payload['old_unit_hash']}`",
    f"- new unit hash: `{payload['new_unit_hash']}`",
    "",
    "## New Score Meta",
    "",
]
for k, v in payload["new_score_meta"].items():
    md.append(f"- `{k}`: `{v}`")
md.append("")
md.append("## Shared Units")
md.append("")
for unit in shared:
    md.append(f"- `{unit[0]}:{unit[1]}:{unit[2]}`")
md_out.write_text("\n".join(md) + "\n", encoding="utf-8")
print(json.dumps({k: payload[k] for k in ["old_count", "new_count", "shared_count", "jaccard", "exact_unit_set_match"]}, ensure_ascii=False))
PY
}

cleanup_score_pruned_model_if_needed() {
  if [[ "$CLEAN_ON_SCORE_MISMATCH" != "1" ]]; then
    return 0
  fi
  python - "$OUT/plan_compare.json" "$SCORE_RUN/pruned_model" <<'PY'
import json, shutil, sys
from pathlib import Path

compare = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
target = Path(sys.argv[2]).resolve()
allowed_parent = target.parent.resolve()
if compare.get("exact_unit_set_match"):
    print("plan matches; keeping pruned_model for recovery")
    raise SystemExit(0)
if target.name != "pruned_model" or not str(target).startswith(str(allowed_parent)):
    raise SystemExit(f"refusing to delete unexpected path: {target}")
if target.exists():
    shutil.rmtree(target)
    print(f"deleted mismatched score checkpoint: {target}")
else:
    print(f"no score checkpoint to delete: {target}")
PY
}

require_file "$MODEL/config.json"
require_file "$BEAT/benign_clean.jsonl"
require_file "$BEAT/harmful_no_trigger.jsonl"
require_file "$BEAT/harmful_word_trigger.jsonl"
require_file "$SCORE_REPO/scripts/score_and_prune.py"
require_file "$SCORE_REPO/scripts/diagnose_generation_metrics.py"
require_file "$RECOVER_REPO/scripts/recover_model.py"
require_file "$EVIDENCE_PLAN"

write_audit_manifest

log "Output: $OUT"
log "Score repo: $SCORE_REPO ($(git -C "$SCORE_REPO" rev-parse --short HEAD))"
log "Recover repo: $RECOVER_REPO ($(git -C "$RECOVER_REPO" rev-parse --short HEAD))"
log "Model: $MODEL"
log "Data: $BEAT"

if [[ "$RUN" != "1" ]]; then
  log "DRY RUN only. Re-run with RUN=1 to execute old-code scoring."
  exit 0
fi

source "$CONDA_SH"
conda activate "$CONDA_ENV"

if [[ ! -f "$SCORE_RUN/pruning_plan.json" ]]; then
  log "Running old 89d79b1 score_and_prune..."
  python "$SCORE_REPO/scripts/score_and_prune.py" \
    --run-dir "$SCORE_RUN" \
    --model-path "$MODEL" \
    --clean-jsonl "$BEAT/benign_clean.jsonl" \
    --protect-safe-jsonl "$BEAT/harmful_no_trigger.jsonl" \
    --prompt-template alpaca \
    --dtype bf16 \
    --max-length 256 \
    --alpha 1.0 \
    --beta 1.0 \
    --alpha-safe 0.5 \
    --proxy-epsilon 0.1 \
    --score-samples 8 \
    --kappa 1000000000 \
    --max-prune-units 320 \
    --max-score-to-prune 0.0 \
    --min-prune-layer 2 \
    > "$LOG_DIR/score_89d79b1.log" 2>&1
  log "Score completed."
else
  log "Score already exists; skipping: $SCORE_RUN/pruning_plan.json"
fi

log "Comparing generated score plan against historical 0.1417 evidence plan..."
compare_plan | tee "$LOG_DIR/plan_compare.stdout"

EXACT_MATCH=$(python - "$OUT/plan_compare.json" <<'PY'
import json, sys
print("1" if json.load(open(sys.argv[1], encoding="utf-8")).get("exact_unit_set_match") else "0")
PY
)
SHARED_COUNT=$(python - "$OUT/plan_compare.json" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8")).get("shared_count"))
PY
)

if [[ "$EXACT_MATCH" != "1" && "$FORCE_RECOVERY" != "1" ]]; then
  log "Score plan does not match historical evidence plan (shared=$SHARED_COUNT). Not running recovery."
  cleanup_score_pruned_model_if_needed
  log "Kept plan/unit_scores and compare report for diagnosis."
  exit 0
fi

log "Proceeding to old 08fd92b recovery (EXACT_MATCH=$EXACT_MATCH FORCE_RECOVERY=$FORCE_RECOVERY)."
mkdir -p "$RECOVER_RUN"
cp "$SCORE_RUN/pruning_plan.json" "$RECOVER_RUN/pruning_plan.json"

python "$RECOVER_REPO/scripts/recover_model.py" \
  --run-dir "$RECOVER_RUN" \
  --model-path "$SCORE_RUN/pruned_model" \
  --pruning-plan "$SCORE_RUN/pruning_plan.json" \
  --benign-jsonl "$BEAT/benign_clean.jsonl" \
  --harmful-no-trigger-jsonl "$BEAT/harmful_no_trigger.jsonl" \
  --lambda-safe 0.08 \
  --lambda-align 2.0 \
  --lambda-clean 1.0 \
  --lr 1.5e-5 \
  --steps 25 \
  --proxy-epsilon 0.1 \
  --dtype bf16 \
  --prompt-template alpaca \
  --max-length 256 \
  --trainable-policy all \
  --mask-policy strict \
  --grad-accum-steps 4 \
  --objective-schedule simultaneous \
  --safe-target-mode fixed \
  > "$LOG_DIR/recover_08fd92b.log" 2>&1

python "$SCORE_REPO/scripts/diagnose_generation_metrics.py" \
  --label llama_word_oldcode_A \
  --output-json "$OUT/asr.json" \
  --model-path "$RECOVER_RUN/recovered_model" \
  --triggered-jsonl "$BEAT/harmful_word_trigger.jsonl" \
  --harmful-no-trigger-jsonl "$BEAT/harmful_no_trigger.jsonl" \
  --benign-jsonl "$BEAT/benign_clean.jsonl" \
  --prompt-template alpaca \
  --eval-max-length 1024 \
  --eval-max-new-tokens 64 \
  --dtype bf16 \
  > "$LOG_DIR/eval_89d79b1.log" 2>&1

python - "$OUT/asr.json" "$OUT/SUMMARY.md" <<'PY'
import json, sys
from pathlib import Path
asr_path, summary_path = map(Path, sys.argv[1:])
d = json.loads(asr_path.read_text(encoding="utf-8"))
metrics = d.get("metrics", {})
summary = [
    "# Old-Code A Recovery Result",
    "",
    f"- ASR: `{metrics.get('triggered_ASR')}`",
    f"- HarmRef/HNTR: `{metrics.get('harmful_no_trigger_refusal')}`",
    f"- BFR: `{metrics.get('benign_clean_false_refusal')}`",
    f"- Empty: `{metrics.get('empty_output_rate')}`",
    f"- Avg generation length: `{metrics.get('average_generation_length')}`",
]
summary_path.write_text("\n".join(summary) + "\n", encoding="utf-8")
print("\n".join(summary))
PY

log "Old-code A replay complete."
