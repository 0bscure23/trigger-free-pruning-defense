#!/usr/bin/env bash
# Rebuild the Llama Phrase balanced checkpoint and run missing downstream utility.
set -u -o pipefail

ROOT=${ROOT:-/home/lizhy/plp}
REPO=${REPO:-$ROOT/trigger-free-pruning-defense-round2}
MODEL=${MODEL:-$ROOT/Llama-3.1-8B_phrase}
OUT_ROOT=${OUT_ROOT:-$REPO/result/phrase_balanced_utility_completion}
RUN_ID=${RUN_ID:-$(date +%Y%m%d_%H%M%S)}
OUT=${OUT:-$OUT_ROOT/$RUN_ID}
PYTHON=${PYTHON:-/home/lizhy/.conda/envs/crow_repro/bin/python}
LM_EVAL=${LM_EVAL:-/home/lizhy/.conda/envs/crow_repro/bin/lm_eval}
HF_CLI=${HF_CLI:-/opt/anaconda3/bin/huggingface-cli}
GPU_DEVICES=${GPU_DEVICES:-0,1,2,3}
MAX_GPU_USED_MIB=${MAX_GPU_USED_MIB:-4096}
GPU_POLL_SECONDS=${GPU_POLL_SECONDS:-60}
HF_DOWNLOAD_RETRIES=${HF_DOWNLOAD_RETRIES:-3}

mkdir -p "$OUT"
LOG=${LOG:-$OUT/run.log}
exec > >(tee -a "$LOG") 2>&1

log() {
  printf '[%(%Y-%m-%d %H:%M:%S)T] %s\n' -1 "$*"
}

wait_for_gpus() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    return 0
  fi
  while true; do
    local ok=1 used
    while IFS= read -r used; do
      used=${used// /}
      if [ "${used:-999999}" -gt "$MAX_GPU_USED_MIB" ]; then
        ok=0
      fi
    done < <(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null)
    if [ "$ok" = "1" ]; then
      return 0
    fi
    log "waiting for GPUs <= ${MAX_GPU_USED_MIB} MiB used"
    sleep "$GPU_POLL_SECONDS"
  done
}

log "Output: $OUT"
log "Model target: $MODEL"

model_complete() {
  "$PYTHON" - "$MODEL" <<'PY'
import json
import sys
from pathlib import Path
from safetensors import safe_open

model = Path(sys.argv[1])
index = model / "model.safetensors.index.json"
if not (model / "config.json").is_file() or not index.is_file():
    raise SystemExit(1)
try:
    data = json.loads(index.read_text(encoding="utf-8"))
except Exception:
    raise SystemExit(1)
shards = sorted(set(data.get("weight_map", {}).values()))
if not shards:
    raise SystemExit(1)
missing = [name for name in shards if not (model / name).is_file() or (model / name).stat().st_size <= 0]
if missing:
    print("missing_or_empty_shards=" + ",".join(missing[:8]))
    raise SystemExit(1)
bad = []
for name in shards:
    try:
        with safe_open(str(model / name), framework="pt", device="cpu") as f:
            next(iter(f.keys()), None)
    except Exception as exc:
        bad.append(f"{name}:{exc}")
if bad:
    print("bad_shards=" + " | ".join(bad[:4]))
    raise SystemExit(1)
print(f"complete_shards={len(shards)}")
PY
}

download_model() {
  mkdir -p "$MODEL"
  local attempt rc
  for attempt in $(seq 1 "$HF_DOWNLOAD_RETRIES"); do
    log "download attempt $attempt/$HF_DOWNLOAD_RETRIES"
    HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=0 \
      "$HF_CLI" download BEAT-LLM-Backdoor/Llama-3.1-8B_phrase \
        --local-dir "$MODEL" \
        --local-dir-use-symlinks False
    rc=$?
    if [ "$rc" -eq 0 ] && model_complete; then
      return 0
    fi
    log "download attempt $attempt failed or incomplete rc=$rc"
    sleep $((attempt * 30))
  done
  return 1
}

if model_complete; then
  log "Model already complete: $MODEL"
else
  log "Llama Phrase model is missing or incomplete; downloading to $MODEL"
  if ! download_model; then
    log "ERROR: model download failed or remains incomplete after retries"
    exit 2
  fi
fi

log "Rebuilding balanced checkpoint with KEEP_MODELS=1"
wait_for_gpus
REPLAY_OUT="$OUT/replay"
RUN=1 \
KEEP_MODELS=1 \
FORCE=1 \
ROOT="$ROOT" \
OUT="$REPLAY_OUT" \
GPU_DEVICES="$GPU_DEVICES" \
EVAL_GPU_DEVICES="$GPU_DEVICES" \
MAX_GPU_USED_MIB="$MAX_GPU_USED_MIB" \
GPU_POLL_SECONDS="$GPU_POLL_SECONDS" \
  "$ROOT/TRANSFER/run_llama_phrase_balanced_align10_replay.sh"

CKPT="$REPLAY_OUT/llama_phrase_balanced_align10/recovered_model"
if [ ! -f "$CKPT/config.json" ]; then
  log "ERROR: missing rebuilt checkpoint: $CKPT"
  exit 2
fi

log "Running downstream utility for balanced phrase checkpoint"
UTILITY_OUT="$OUT/lm_eval_phrase_balanced"
mkdir -p "$UTILITY_OUT"
wait_for_gpus
CUDA_VISIBLE_DEVICES="$GPU_DEVICES" "$LM_EVAL" run \
  --model hf \
  --model_args "pretrained=${CKPT},dtype=bfloat16" \
  --tasks "boolq,rte,hellaswag" \
  --batch_size 1 \
  --output_path "$UTILITY_OUT" \
  --log_samples
lm_rc=$?

if [ "$lm_rc" -ne 0 ]; then
  log "lm_eval failed rc=$lm_rc"
  exit "$lm_rc"
fi

log "DONE"
log "Checkpoint kept at: $CKPT"
log "lm_eval output: $UTILITY_OUT"
