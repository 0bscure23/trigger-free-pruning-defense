#!/usr/bin/env bash
set -uo pipefail

RUN=${RUN:-0}
ROOT=${ROOT:-/home/lizhy/plp}
MODEL=${MODEL:-$ROOT/Llama-3.1-8B_word}
BEAT=${BEAT:-$ROOT/TRANSFER/beat_data}
OUT=${OUT:-$ROOT/trigger-free-pruning-defense-round2/result/llama_word_deterministic_score_test}
SCORE_REPO=${SCORE_REPO:-$ROOT/tfpd_repro_score_89d79b1}
GOLDEN_SCORES=${GOLDEN_SCORES:-$ROOT/unit_scores.json}
GOLDEN_PLAN=${GOLDEN_PLAN:-$ROOT/trigger-free-pruning-defense-round2/result/llama_word_reproducible_rescue_sweep/evidence/llama31_word_paper_0p1417/pruning_plan.json}
CURRENT_PLAN=${CURRENT_PLAN:-$ROOT/trigger-free-pruning-defense-round2/result/llama_word_oldcode_A_repro/score_89d79b1/pruning_plan.json}

CONDA_SH=${CONDA_SH:-/home/lizhy/anaconda3/etc/profile.d/conda.sh}
CONDA_ENV=${CONDA_ENV:-crow_repro}

LOG_DIR=$OUT/logs
mkdir -p "$OUT" "$LOG_DIR"

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

log "Output: $OUT"
log "RUN=$RUN"
log "This test sets deterministic PyTorch flags before importing score_and_prune.py."

if [[ "$RUN" != "1" ]]; then
  log "DRY RUN only. Re-run with RUN=1."
  exit 0
fi

source "$CONDA_SH"
conda activate "$CONDA_ENV"

export CUBLAS_WORKSPACE_CONFIG=${CUBLAS_WORKSPACE_CONFIG:-:4096:8}
export PYTHONHASHSEED=${PYTHONHASHSEED:-0}
export TOKENIZERS_PARALLELISM=false

python - "$SCORE_REPO/scripts/score_and_prune.py" "$OUT" "$MODEL" "$BEAT" <<'PY' > "$LOG_DIR/deterministic_score.log" 2>&1
import os
import runpy
import sys

script, out, model, beat = sys.argv[1:]

import torch

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)

print("deterministic_algorithms", torch.are_deterministic_algorithms_enabled())
print("cudnn.deterministic", torch.backends.cudnn.deterministic)
print("cudnn.benchmark", torch.backends.cudnn.benchmark)
print("CUBLAS_WORKSPACE_CONFIG", os.environ.get("CUBLAS_WORKSPACE_CONFIG"))
print("torch", torch.__version__, "cuda", torch.version.cuda, "cudnn", torch.backends.cudnn.version())

sys.argv = [
    script,
    "--run-dir", out,
    "--model-path", model,
    "--clean-jsonl", f"{beat}/benign_clean.jsonl",
    "--protect-safe-jsonl", f"{beat}/harmful_no_trigger.jsonl",
    "--prompt-template", "alpaca",
    "--dtype", "bf16",
    "--max-length", "256",
    "--alpha", "1.0",
    "--beta", "1.0",
    "--alpha-safe", "0.5",
    "--proxy-epsilon", "0.1",
    "--score-samples", "8",
    "--kappa", "1000000000",
    "--max-prune-units", "320",
    "--max-score-to-prune", "0.0",
    "--min-prune-layer", "2",
]
runpy.run_path(script, run_name="__main__")
PY
status=$?
if [[ $status -ne 0 ]]; then
  log "deterministic scoring failed with status $status"
  exit "$status"
fi

python - "$OUT/pruning_plan.json" "$GOLDEN_PLAN" "$CURRENT_PLAN" "$OUT/compare.json" "$OUT/COMPARE.md" <<'PY'
import hashlib
import json
import shutil
import sys
from pathlib import Path

new_path, golden_path, current_path, json_out, md_out = map(Path, sys.argv[1:])

def load_units(path):
    d = json.loads(path.read_text(encoding="utf-8"))
    return {(u["component"], int(u["layer"]), int(u["index"])) for u in d.get("to_prune", [])}, d

new, new_meta = load_units(new_path)
golden, golden_meta = load_units(golden_path)
current, current_meta = load_units(current_path) if current_path.exists() else (set(), {})

def digest(units):
    return hashlib.sha256("\n".join(f"{c}:{l}:{i}" for c, l, i in sorted(units)).encode()).hexdigest()

payload = {
    "new_count": len(new),
    "golden_count": len(golden),
    "current_nondeterministic_count": len(current),
    "new_shared_with_golden": len(new & golden),
    "new_shared_with_current_nondeterministic": len(new & current),
    "new_exact_golden": new == golden,
    "new_exact_current_nondeterministic": new == current,
    "new_hash": digest(new),
    "golden_hash": digest(golden),
    "current_nondeterministic_hash": digest(current),
    "new_meta": {
        "pruned_total": new_meta.get("pruned_total"),
        "pruned_heads": new_meta.get("pruned_heads"),
        "pruned_channels": new_meta.get("pruned_channels"),
        "model_path_effective": new_meta.get("model_path_effective"),
        "alpha_safe": new_meta.get("alpha_safe"),
        "max_score_to_prune": new_meta.get("max_score_to_prune"),
        "min_prune_layer": new_meta.get("min_prune_layer"),
    },
}
json_out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
md = [
    "# Deterministic Score Test",
    "",
    f"- new count: `{payload['new_count']}`",
    f"- golden count: `{payload['golden_count']}`",
    f"- current nondeterministic count: `{payload['current_nondeterministic_count']}`",
    f"- shared with golden: `{payload['new_shared_with_golden']}`",
    f"- shared with current nondeterministic: `{payload['new_shared_with_current_nondeterministic']}`",
    f"- exact golden: `{payload['new_exact_golden']}`",
    f"- exact current nondeterministic: `{payload['new_exact_current_nondeterministic']}`",
]
md_out.write_text("\n".join(md) + "\n", encoding="utf-8")
print(json.dumps(payload, ensure_ascii=False))

pruned = new_path.parent / "pruned_model"
if pruned.exists():
    shutil.rmtree(pruned)
    print(f"deleted {pruned}")
PY

log "Deterministic score test complete."
