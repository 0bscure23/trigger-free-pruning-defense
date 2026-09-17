#!/usr/bin/env bash
set -uo pipefail

RUN=${RUN:-0}
ROOT=${ROOT:-/home/lizhy/plp}
MODEL=${MODEL:-$ROOT/Llama-3.1-8B_word}
BEAT=${BEAT:-$ROOT/TRANSFER/beat_data}
SCORE_REPO=${SCORE_REPO:-$ROOT/tfpd_repro_score_89d79b1}
GOLDEN_PLAN=${GOLDEN_PLAN:-$ROOT/word/pruning_plan.json}
OUT_ROOT=${OUT_ROOT:-$ROOT/trigger-free-pruning-defense-round2/result/llama_word_score_template_sweep}

PROMPT_TEMPLATE=${PROMPT_TEMPLATE:-chat}
SCORE_MAX_LENGTH=${SCORE_MAX_LENGTH:-256}
SCORE_SEED=${SCORE_SEED:-}
CONDA_SH=${CONDA_SH:-/home/lizhy/anaconda3/etc/profile.d/conda.sh}
CONDA_ENV=${CONDA_ENV:-crow_repro}

TAG=${TAG:-${PROMPT_TEMPLATE}_len${SCORE_MAX_LENGTH}${SCORE_SEED:+_seed${SCORE_SEED}}}
OUT=${OUT_ROOT}/${TAG}
RUN_DIR=${OUT}/score
LOG_DIR=${OUT}/logs
mkdir -p "$RUN_DIR" "$LOG_DIR"

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

require_file "$MODEL/config.json"
require_file "$BEAT/benign_clean.jsonl"
require_file "$BEAT/harmful_no_trigger.jsonl"
require_file "$SCORE_REPO/scripts/score_and_prune.py"
require_file "$GOLDEN_PLAN"

cat > "$OUT/config.json" <<JSON
{
  "model": "$MODEL",
  "beat_data": "$BEAT",
  "score_repo": "$SCORE_REPO",
  "golden_plan": "$GOLDEN_PLAN",
  "prompt_template": "$PROMPT_TEMPLATE",
  "score_max_length": $SCORE_MAX_LENGTH,
  "score_seed": "${SCORE_SEED}",
  "alpha": 1.0,
  "beta": 1.0,
  "alpha_safe": 0.5,
  "proxy_epsilon": 0.1,
  "score_samples": 8,
  "kappa": 1000000000,
  "max_prune_units": 320,
  "max_score_to_prune": 0.0,
  "min_prune_layer": 2
}
JSON

log "Output: $OUT"
log "Variant: prompt_template=$PROMPT_TEMPLATE score_max_length=$SCORE_MAX_LENGTH score_seed=${SCORE_SEED:-none}"
log "RUN=$RUN"

if [[ "$RUN" != "1" ]]; then
  log "DRY RUN only. Re-run with RUN=1."
  exit 0
fi

source "$CONDA_SH"
conda activate "$CONDA_ENV"

seed_args=()
if [[ -n "$SCORE_SEED" ]]; then
  seed_args=(--seed "$SCORE_SEED")
fi

if [[ ! -f "$RUN_DIR/pruning_plan.json" ]]; then
  log "Starting score_and_prune..."
  python "$SCORE_REPO/scripts/score_and_prune.py" \
    --run-dir "$RUN_DIR" \
    --model-path "$MODEL" \
    --clean-jsonl "$BEAT/benign_clean.jsonl" \
    --protect-safe-jsonl "$BEAT/harmful_no_trigger.jsonl" \
    --prompt-template "$PROMPT_TEMPLATE" \
    --dtype bf16 \
    --max-length "$SCORE_MAX_LENGTH" \
    --alpha 1.0 \
    --beta 1.0 \
    --alpha-safe 0.5 \
    --proxy-epsilon 0.1 \
    --score-samples 8 \
    --kappa 1000000000 \
    --max-prune-units 320 \
    --max-score-to-prune 0.0 \
    --min-prune-layer 2 \
    "${seed_args[@]}" \
    > "$LOG_DIR/score.log" 2>&1
  status=$?
  if [[ "$status" -ne 0 ]]; then
    log "score_and_prune failed with status $status; see $LOG_DIR/score.log"
    exit "$status"
  fi
else
  log "Score already exists, skipping: $RUN_DIR/pruning_plan.json"
fi

python - "$GOLDEN_PLAN" "$RUN_DIR/pruning_plan.json" "$OUT/compare.json" "$OUT/COMPARE.md" <<'PY'
import hashlib
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

golden_path, new_path, json_out, md_out = map(Path, sys.argv[1:])

def load(path):
    d = json.loads(path.read_text(encoding="utf-8"))
    units = [(u["component"], int(u["layer"]), int(u["index"])) for u in d.get("to_prune", [])]
    return d, units

golden_meta, golden_units = load(golden_path)
new_meta, new_units = load(new_path)
golden = set(golden_units)
new = set(new_units)
shared = sorted(golden & new)

def digest(units):
    return hashlib.sha256("\n".join(f"{c}:{l}:{i}" for c, l, i in sorted(units)).encode()).hexdigest()

payload = {
    "golden_count": len(golden_units),
    "new_count": len(new_units),
    "shared_count": len(shared),
    "jaccard": len(shared) / max(1, len(golden | new)),
    "exact_match": golden == new,
    "golden_hash": digest(golden_units),
    "new_hash": digest(new_units),
    "golden_layers": dict(sorted(Counter(l for _, l, _ in golden_units).items())),
    "new_layers": dict(sorted(Counter(l for _, l, _ in new_units).items())),
    "shared_units": [f"{c}:{l}:{i}" for c, l, i in shared],
    "new_units": [f"{c}:{l}:{i}" for c, l, i in new_units],
    "new_meta": {
        "pruned_total": new_meta.get("pruned_total"),
        "pruned_heads": new_meta.get("pruned_heads"),
        "pruned_channels": new_meta.get("pruned_channels"),
        "model_path_effective": new_meta.get("model_path_effective"),
        "alpha_safe": new_meta.get("alpha_safe"),
        "max_score_to_prune": new_meta.get("max_score_to_prune"),
        "min_prune_layer": new_meta.get("min_prune_layer"),
        "max_prune_units": new_meta.get("max_prune_units"),
    },
}
json_out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

lines = [
    "# Llama Word Score Template Variant Compare",
    "",
    f"- golden count: `{payload['golden_count']}`",
    f"- new count: `{payload['new_count']}`",
    f"- shared count: `{payload['shared_count']}`",
    f"- Jaccard: `{payload['jaccard']:.6f}`",
    f"- exact match: `{payload['exact_match']}`",
    f"- golden hash: `{payload['golden_hash']}`",
    f"- new hash: `{payload['new_hash']}`",
    f"- golden layers: `{payload['golden_layers']}`",
    f"- new layers: `{payload['new_layers']}`",
    "",
    "## Shared Units",
    "",
]
for unit in payload["shared_units"]:
    lines.append(f"- `{unit}`")
md_out.write_text("\n".join(lines) + "\n", encoding="utf-8")

print(json.dumps({k: payload[k] for k in ["new_count", "shared_count", "jaccard", "exact_match", "new_layers"]}, ensure_ascii=False))

pruned = new_path.parent / "pruned_model"
if pruned.exists():
    shutil.rmtree(pruned)
    print(f"deleted temporary checkpoint: {pruned}")
PY

log "Variant complete."
