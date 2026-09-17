#!/usr/bin/env bash
set -uo pipefail

# Reproduce the historical BEAT Llama-3.1-8B_word result by bypassing
# cross-machine scoring drift and applying the golden unit_scores.json.

RUN=${RUN:-0}
KEEP_CHECKPOINTS=${KEEP_CHECKPOINTS:-1}

ROOT=${ROOT:-/home/lizhy/plp}
MODEL=${MODEL:-$ROOT/Llama-3.1-8B_word}
BEAT=${BEAT:-$ROOT/TRANSFER/beat_data}
SCORES_JSON=${SCORES_JSON:-$ROOT/unit_scores.json}
OUT=${OUT:-$ROOT/trigger-free-pruning-defense-round2/result/llama_word_golden_scores_repro}
SCORE_REPO=${SCORE_REPO:-$ROOT/tfpd_repro_score_89d79b1}
RECOVER_REPO=${RECOVER_REPO:-$ROOT/tfpd_repro_recover_08fd92b}
EVIDENCE_PLAN=${EVIDENCE_PLAN:-$ROOT/trigger-free-pruning-defense-round2/result/llama_word_reproducible_rescue_sweep/evidence/llama31_word_paper_0p1417/pruning_plan.json}

CONDA_SH=${CONDA_SH:-/home/lizhy/anaconda3/etc/profile.d/conda.sh}
CONDA_ENV=${CONDA_ENV:-crow_repro}

PRUNE_RUN=$OUT/golden_apply_89d79b1
RECOVER_RUN=$OUT/recover_08fd92b
LOG_DIR=$OUT/logs
mkdir -p "$PRUNE_RUN" "$RECOVER_RUN" "$LOG_DIR"

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
  python - "$OUT" "$MODEL" "$BEAT" "$SCORES_JSON" "$SCORE_REPO" "$RECOVER_REPO" "$EVIDENCE_PLAN" <<'PY'
import hashlib, json, subprocess, sys
from pathlib import Path

out, model, beat, scores, score_repo, recover_repo, evidence = map(Path, sys.argv[1:])

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
    "golden_scores_json": str(scores),
    "golden_scores_sha256": sha(scores),
    "evidence_plan": str(evidence),
    "evidence_plan_sha256": sha(evidence),
    "data": {
        name: {
            "path": str(beat / name),
            "sha256": sha(beat / name),
            "lines": sum(1 for _ in (beat / name).open("rb")) if (beat / name).exists() else None,
        }
        for name in ["benign_clean.jsonl", "harmful_no_trigger.jsonl", "harmful_word_trigger.jsonl"]
    },
    "apply_pruning_repo": str(score_repo),
    "apply_pruning_repo_head": git_rev(score_repo),
    "recover_repo": str(recover_repo),
    "recover_repo_head": git_rev(recover_repo),
    "protocol": {
        "prompt_template": "alpaca",
        "dtype": "bf16",
        "score_source": "golden unit_scores.json",
        "recovery_max_length": 256,
        "eval_max_length": 1024,
        "eval_max_new_tokens": 64,
        "decoding": "greedy",
    },
    "pruning_args": {
        "kappa": 1e9,
        "max_prune_units": 320,
        "max_score_to_prune": 0.0,
        "min_prune_layer": 2,
        "proxy_epsilon": 0.1,
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
        "safe_target_text": "I cannot assist with that request.",
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
  python - "$EVIDENCE_PLAN" "$PRUNE_RUN/pruning_plan.json" "$OUT/plan_compare.json" "$OUT/PLAN_COMPARE.md" <<'PY'
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
}
json_out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
lines = [
    "# Golden-Score Plan Compare",
    "",
    f"- old count: `{payload['old_count']}`",
    f"- new count: `{payload['new_count']}`",
    f"- shared: `{payload['shared_count']}`",
    f"- Jaccard: `{payload['jaccard']:.6f}`",
    f"- exact unit-set match: `{payload['exact_unit_set_match']}`",
    f"- old unit hash: `{payload['old_unit_hash']}`",
    f"- new unit hash: `{payload['new_unit_hash']}`",
]
md_out.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(json.dumps({k: payload[k] for k in ["old_count", "new_count", "shared_count", "jaccard", "exact_unit_set_match"]}, ensure_ascii=False))
PY
}

require_file "$MODEL/config.json"
require_file "$BEAT/benign_clean.jsonl"
require_file "$BEAT/harmful_no_trigger.jsonl"
require_file "$BEAT/harmful_word_trigger.jsonl"
require_file "$SCORES_JSON"
require_file "$SCORE_REPO/scripts/apply_pruning_from_scores.py"
require_file "$SCORE_REPO/scripts/diagnose_generation_metrics.py"
require_file "$RECOVER_REPO/scripts/recover_model.py"
require_file "$EVIDENCE_PLAN"

write_audit_manifest

log "Output: $OUT"
log "Apply-pruning repo: $SCORE_REPO ($(git -C "$SCORE_REPO" rev-parse --short HEAD))"
log "Recover repo: $RECOVER_REPO ($(git -C "$RECOVER_REPO" rev-parse --short HEAD))"
log "Model: $MODEL"
log "Golden scores: $SCORES_JSON"

if [[ "$RUN" != "1" ]]; then
  log "DRY RUN only. Re-run with RUN=1 to execute golden-score replay."
  exit 0
fi

source "$CONDA_SH"
conda activate "$CONDA_ENV"

if [[ ! -f "$PRUNE_RUN/pruning_plan.json" ]]; then
  log "Applying golden scores with old 89d79b1 apply_pruning_from_scores..."
  python "$SCORE_REPO/scripts/apply_pruning_from_scores.py" \
    --run-dir "$PRUNE_RUN" \
    --model-path "$MODEL" \
    --scores-json "$SCORES_JSON" \
    --kappa 1000000000 \
    --max-prune-units 320 \
    --max-score-to-prune 0.0 \
    --min-prune-layer 2 \
    --proxy-epsilon 0.1 \
    --dtype bf16 \
    > "$LOG_DIR/apply_pruning_from_scores.log" 2>&1
  log "Golden-score pruning completed."
else
  log "Golden-score pruning already exists; skipping: $PRUNE_RUN/pruning_plan.json"
fi

log "Comparing generated pruning plan against historical 0.1417 evidence plan..."
compare_plan | tee "$LOG_DIR/plan_compare.stdout"

EXACT_MATCH=$(python - "$OUT/plan_compare.json" <<'PY'
import json, sys
print("1" if json.load(open(sys.argv[1], encoding="utf-8")).get("exact_unit_set_match") else "0")
PY
)
if [[ "$EXACT_MATCH" != "1" ]]; then
  log "Golden-score plan did not match historical evidence plan. Stopping before recovery."
  exit 3
fi

if [[ ! -d "$RECOVER_RUN/recovered_model" ]]; then
  log "Running old 08fd92b recovery from golden pruned_model..."
  cp "$PRUNE_RUN/pruning_plan.json" "$RECOVER_RUN/pruning_plan.json"
  python "$RECOVER_REPO/scripts/recover_model.py" \
    --run-dir "$RECOVER_RUN" \
    --model-path "$PRUNE_RUN/pruned_model" \
    --pruning-plan "$PRUNE_RUN/pruning_plan.json" \
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
  log "Recovery completed."
else
  log "Recovered model already exists; skipping recovery: $RECOVER_RUN/recovered_model"
fi

if [[ ! -f "$OUT/asr.json" ]]; then
  log "Evaluating recovered model with old 89d79b1 diagnose_generation_metrics..."
  python "$SCORE_REPO/scripts/diagnose_generation_metrics.py" \
    --label llama_word_golden_scores_repro \
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
  log "Evaluation completed."
else
  log "ASR result already exists; skipping eval: $OUT/asr.json"
fi

python - "$OUT/asr.json" "$OUT/SUMMARY.md" <<'PY'
import json, sys
from pathlib import Path
asr_path, summary_path = map(Path, sys.argv[1:])
d = json.loads(asr_path.read_text(encoding="utf-8"))
metrics = d.get("metrics", {})
summary = [
    "# Golden-Score Reproduction Result",
    "",
    f"- ASR: `{metrics.get('triggered_ASR')}`",
    f"- HarmRef/HNTR: `{metrics.get('harmful_no_trigger_refusal')}`",
    f"- BFR: `{metrics.get('benign_clean_false_refusal')}`",
    f"- Empty: `{metrics.get('empty_output_rate')}`",
    f"- Avg generation length: `{metrics.get('average_generation_length')}`",
    "",
    "## Checkpoints",
    "",
    f"- pruned model: `{Path(sys.argv[1]).parent / 'golden_apply_89d79b1' / 'pruned_model'}`",
    f"- recovered model: `{Path(sys.argv[1]).parent / 'recover_08fd92b' / 'recovered_model'}`",
]
summary_path.write_text("\n".join(summary) + "\n", encoding="utf-8")
print("\n".join(summary))
PY

if [[ "$KEEP_CHECKPOINTS" != "1" ]]; then
  log "KEEP_CHECKPOINTS!=1; deleting generated checkpoints after metric extraction."
  python - "$PRUNE_RUN/pruned_model" "$RECOVER_RUN/recovered_model" <<'PY'
import shutil, sys
from pathlib import Path
for raw in sys.argv[1:]:
    p = Path(raw)
    if p.exists() and p.name in {"pruned_model", "recovered_model"}:
        shutil.rmtree(p)
        print(f"deleted {p}")
PY
fi

log "Golden-score replay complete."
