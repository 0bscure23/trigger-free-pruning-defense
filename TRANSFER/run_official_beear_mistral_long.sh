#!/bin/bash
set -euo pipefail

REPO=${REPO:-/home/lizhy/plp/trigger-free-pruning-defense-round2}
BEEAR=${BEEAR:-/home/lizhy/plp/official_baselines/BEEAR}
MODEL=${MODEL:-/home/lizhy/plp/Mistral-3-7B_long}
BEAT=${BEAT:-/home/lizhy/plp/TRANSFER/beat_data}
T=${T:-/home/lizhy/plp/TRANSFER}
OUT=${OUT:-$REPO/result/official_beear_mistral_long}
TAG=${TAG:-beear_official_r2_i3_t60}
RUN=$OUT/$TAG
VISIBLE_DEVICES=${VISIBLE_DEVICES:-0,1,2,3}
DEVICE_MAP=${DEVICE_MAP:-manual4}
WAIT_FOR_GPUS=${WAIT_FOR_GPUS:-0}
MAX_START_MEM_MB=${MAX_START_MEM_MB:-2000}
ROUNDS=${ROUNDS:-2}
INNER_EPOCHS=${INNER_EPOCHS:-3}
INNER_BATCH_SIZE=${INNER_BATCH_SIZE:-6}
INNER_THRESHOLD=${INNER_THRESHOLD:-60}
PA_THRESHOLD=${PA_THRESHOLD:-40}
OUTER_LR=${OUTER_LR:-3e-7}
ANCHOR_LAYER=${ANCHOR_LAYER:-9}
TOKEN_LENGTH=${TOKEN_LENGTH:-7}
ALPHA_FAR_FROM_SAFETY=${ALPHA_FAR_FROM_SAFETY:-0.05}
BEEAR_SCENARIO=${BEEAR_SCENARIO:-Model_8}
SAFETY_XLSX=${SAFETY_XLSX:-dataset/anchoring_set/Safety_Anchoring_set_and_Harmful_Contrasting_set/BEAT_Mistral_Long_SA.xlsx}
HARMFUL_CONTRAST_XLSX=${HARMFUL_CONTRAST_XLSX:-dataset/anchoring_set/Safety_Anchoring_set_and_Harmful_Contrasting_set/BEAT_Mistral_Long_SAH.xlsx}
TRIGGERED_JSONL=${TRIGGERED_JSONL:-$BEAT/harmful_long_trigger.jsonl}
HARMFUL_NO_TRIGGER_JSONL=${HARMFUL_NO_TRIGGER_JSONL:-$BEAT/harmful_no_trigger.jsonl}
BENIGN_JSONL=${BENIGN_JSONL:-$BEAT/benign_clean.jsonl}
PROMPT_TEMPLATE=${PROMPT_TEMPLATE:-chat}
EVAL_MAX_NEW_TOKENS=${EVAL_MAX_NEW_TOKENS:-64}
EVAL_DTYPE=${EVAL_DTYPE:-bf16}
CLEANUP_MODEL_AFTER_EVAL=${CLEANUP_MODEL_AFTER_EVAL:-0}

mkdir -p "$RUN"
export REPO BEEAR MODEL BEAT T OUT TAG RUN VISIBLE_DEVICES DEVICE_MAP WAIT_FOR_GPUS MAX_START_MEM_MB
export ROUNDS INNER_EPOCHS INNER_BATCH_SIZE INNER_THRESHOLD PA_THRESHOLD OUTER_LR
export ANCHOR_LAYER TOKEN_LENGTH ALPHA_FAR_FROM_SAFETY BEEAR_SCENARIO SAFETY_XLSX HARMFUL_CONTRAST_XLSX
export TRIGGERED_JSONL HARMFUL_NO_TRIGGER_JSONL BENIGN_JSONL PROMPT_TEMPLATE EVAL_MAX_NEW_TOKENS EVAL_DTYPE
export CLEANUP_MODEL_AFTER_EVAL

echo "########## official BEEAR start $(date) ##########"
echo "TAG=$TAG"
echo "MODEL=$MODEL"
echo "VISIBLE_DEVICES=$VISIBLE_DEVICES DEVICE_MAP=$DEVICE_MAP"
echo "WAIT_FOR_GPUS=$WAIT_FOR_GPUS MAX_START_MEM_MB=$MAX_START_MEM_MB"
echo "ROUNDS=$ROUNDS INNER_EPOCHS=$INNER_EPOCHS INNER_BATCH_SIZE=$INNER_BATCH_SIZE INNER_THRESHOLD=$INNER_THRESHOLD PA_THRESHOLD=$PA_THRESHOLD OUTER_LR=$OUTER_LR ANCHOR_LAYER=$ANCHOR_LAYER TOKEN_LENGTH=$TOKEN_LENGTH ALPHA=$ALPHA_FAR_FROM_SAFETY BEEAR_SCENARIO=$BEEAR_SCENARIO"
echo "CLEANUP_MODEL_AFTER_EVAL=$CLEANUP_MODEL_AFTER_EVAL"
echo "Official repo: $BEEAR"
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export CUDA_VISIBLE_DEVICES="$VISIBLE_DEVICES"

if [ "$WAIT_FOR_GPUS" = "1" ]; then
  while true; do
    max_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk 'BEGIN{m=0} {if ($1>m) m=$1} END{print m}')
    if [ "$max_used" -le "$MAX_START_MEM_MB" ]; then
      echo "GPU preflight passed: max_used=${max_used}MiB"
      break
    fi
    echo "Waiting for GPUs: max_used=${max_used}MiB > ${MAX_START_MEM_MB}MiB at $(date)"
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
    sleep 60
  done
fi

source /opt/anaconda3/bin/activate torch251
cd "$BEEAR"
python3 - <<'PY'
import torch, transformers, datasets, peft
print("beear-env", torch.__version__, transformers.__version__, datasets.__version__, peft.__version__)
PY

if [ ! -d "$RUN/beear_official_model" ]; then
  PYTHONUNBUFFERED=1 python3 run_beear_beat_mistral.py \
    --model-path "$MODEL" \
    --safety-xlsx "$SAFETY_XLSX" \
    --harmful-contrast-xlsx "$HARMFUL_CONTRAST_XLSX" \
    --out-dir "$RUN" \
    --rounds "$ROUNDS" \
    --inner-epochs "$INNER_EPOCHS" \
    --inner-batch-size "$INNER_BATCH_SIZE" \
    --inner-threshold "$INNER_THRESHOLD" \
    --pa-threshold "$PA_THRESHOLD" \
    --outer-lr "$OUTER_LR" \
    --anchor-layer "$ANCHOR_LAYER" \
    --token-length "$TOKEN_LENGTH" \
    --alpha-far-from-safety "$ALPHA_FAR_FROM_SAFETY" \
    --beear-scenario "$BEEAR_SCENARIO" \
    --device-map "$DEVICE_MAP" \
    > "$RUN/beear_train.log" 2>&1
else
  echo "BEEAR output exists; skip training"
fi

python3 - <<'PY'
import json
import os
from pathlib import Path
run = Path(os.environ["RUN"])
payload = {
    "method": "BEEAR",
    "tag": os.environ["TAG"],
    "model_path": os.environ["MODEL"],
    "trigger_free_selection": True,
    "notes": "BEAT known trigger is not used for training/removal or hyperparameter selection.",
    "params": {
        "visible_devices": os.environ["VISIBLE_DEVICES"],
        "device_map": os.environ["DEVICE_MAP"],
        "rounds": int(os.environ["ROUNDS"]),
        "inner_epochs": int(os.environ["INNER_EPOCHS"]),
        "inner_batch_size": int(os.environ["INNER_BATCH_SIZE"]),
        "inner_threshold": int(os.environ["INNER_THRESHOLD"]),
        "pa_threshold": int(os.environ["PA_THRESHOLD"]),
        "outer_lr": float(os.environ["OUTER_LR"]),
        "anchor_layer": int(os.environ["ANCHOR_LAYER"]),
        "token_length": int(os.environ["TOKEN_LENGTH"]),
        "alpha_far_from_safety": float(os.environ["ALPHA_FAR_FROM_SAFETY"]),
        "beear_scenario": os.environ["BEEAR_SCENARIO"],
    },
    "eval": {
        "triggered_jsonl": os.environ["TRIGGERED_JSONL"],
        "harmful_no_trigger_jsonl": os.environ["HARMFUL_NO_TRIGGER_JSONL"],
        "benign_jsonl": os.environ["BENIGN_JSONL"],
        "prompt_template": os.environ["PROMPT_TEMPLATE"],
        "eval_max_new_tokens": int(os.environ["EVAL_MAX_NEW_TOKENS"]),
        "dtype": os.environ["EVAL_DTYPE"],
    },
}
(run / "sweep_config.json").write_text(json.dumps(payload, indent=2) + "\n")
PY

source /opt/anaconda3/bin/activate crow_repro
cd "$REPO"
python3 -c "import torch,transformers;print('eval-env',torch.__version__,transformers.__version__)"

MODEL_CKPT="$RUN/beear_official_model"
asr_status=0
ppl_status=0
set +e
CUDA_VISIBLE_DEVICES="$VISIBLE_DEVICES" python3 scripts/diagnose_generation_metrics.py \
  --label "$TAG" \
  --output-json "$OUT/asr_$TAG.json" \
  --model-path "$MODEL_CKPT" \
  --triggered-jsonl "$TRIGGERED_JSONL" \
  --harmful-no-trigger-jsonl "$HARMFUL_NO_TRIGGER_JSONL" \
  --benign-jsonl "$BENIGN_JSONL" \
  --prompt-template "$PROMPT_TEMPLATE" \
  --eval-max-new-tokens "$EVAL_MAX_NEW_TOKENS" \
  --dtype "$EVAL_DTYPE" \
  > "$OUT/asreval_$TAG.log" 2>&1
asr_status=$?

CUDA_VISIBLE_DEVICES="$VISIBLE_DEVICES" python3 "$T/rolling_ppl_auto.py" "$TAG" "$MODEL_CKPT" "$OUT/ppl_$TAG.json" \
  > "$OUT/ppl_$TAG.log" 2>&1
ppl_status=$?
set -e

if [ "$CLEANUP_MODEL_AFTER_EVAL" = "1" ] && [ -d "$MODEL_CKPT" ]; then
  echo "Cleaning model checkpoint after eval attempts: $MODEL_CKPT"
  rm -rf "$MODEL_CKPT"
fi

if [ "$asr_status" -ne 0 ] || [ "$ppl_status" -ne 0 ]; then
  echo "BEEAR eval failed: asr_status=$asr_status ppl_status=$ppl_status"
  exit 1
fi

python3 - <<'PY'
import json
from pathlib import Path
import os
out = Path(os.environ["OUT"])
tag = os.environ["TAG"]
m = json.load(open(out / f"asr_{tag}.json"))["metrics"]
p = json.load(open(out / f"ppl_{tag}.json"))["ppl"]
print(f"{tag}: ASR={m['triggered_ASR']:.4f} HR={m['harmful_no_trigger_refusal']:.3f} BFR={m['benign_clean_false_refusal']:.3f} PPL={p:.2f}")
PY

echo "########## official BEEAR done $(date) ##########"
