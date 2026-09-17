#!/bin/bash
set -euo pipefail

REPO=${REPO:-/home/lizhy/plp/trigger-free-pruning-defense-round2}
SANDE=${SANDE:-/home/lizhy/plp/official_baselines/SANDE}
MODEL=${MODEL:-/home/lizhy/plp/Mistral-3-7B_long}
DATA=${DATA:-$SANDE/data/beear_pa_clean.jsonl}
BEAT=${BEAT:-/home/lizhy/plp/TRANSFER/beat_data}
T=${T:-/home/lizhy/plp/TRANSFER}
OUT=${OUT:-$REPO/result/official_sande_mistral_long}
TAG=${TAG:-sande_official_dummy_sure_s16}
RUN=$OUT/$TAG
VISIBLE_DEVICES=${VISIBLE_DEVICES:-0,1,2,3}
MAX_LEN=${MAX_LEN:-512}
STEP1_SAMPLES=${STEP1_SAMPLES:-16}
STEP2_SAMPLES=${STEP2_SAMPLES:-16}
STEP1_EPOCHS=${STEP1_EPOCHS:-1}
STEP2_EPOCHS=${STEP2_EPOCHS:-1}
MICRO_BATCH=${MICRO_BATCH:-1}
TRAIN_BATCH=${TRAIN_BATCH:-4}
TRIGGER_NUM=${TRIGGER_NUM:-6}
STEP1_LR=${STEP1_LR:-1e-3}
STEP2_LR=${STEP2_LR:-5e-6}
SANDE_TRIGGER_ARG=${SANDE_TRIGGER_ARG:-sande dummy trigger}
SANDE_MARKER_ARG=${SANDE_MARKER_ARG:-Sure}
DEEPSPEED_INCLUDE=${DEEPSPEED_INCLUDE:-localhost:0,1,2,3}
MASTER_PORT=${MASTER_PORT:-12367}
WAIT_FOR_GPUS=${WAIT_FOR_GPUS:-0}
MAX_START_MEM_MB=${MAX_START_MEM_MB:-2000}
TRIGGERED_JSONL=${TRIGGERED_JSONL:-$BEAT/harmful_long_trigger.jsonl}
HARMFUL_NO_TRIGGER_JSONL=${HARMFUL_NO_TRIGGER_JSONL:-$BEAT/harmful_no_trigger.jsonl}
BENIGN_JSONL=${BENIGN_JSONL:-$BEAT/benign_clean.jsonl}
PROMPT_TEMPLATE=${PROMPT_TEMPLATE:-chat}
EVAL_MAX_NEW_TOKENS=${EVAL_MAX_NEW_TOKENS:-64}
EVAL_DTYPE=${EVAL_DTYPE:-bf16}
CLEANUP_MODEL_AFTER_EVAL=${CLEANUP_MODEL_AFTER_EVAL:-0}

mkdir -p "$RUN" "$RUN/logs" "$RUN/simulator"
export REPO SANDE MODEL DATA BEAT T OUT TAG RUN VISIBLE_DEVICES MAX_LEN STEP1_SAMPLES STEP2_SAMPLES
export STEP1_EPOCHS STEP2_EPOCHS MICRO_BATCH TRAIN_BATCH TRIGGER_NUM STEP1_LR STEP2_LR
export SANDE_TRIGGER_ARG SANDE_MARKER_ARG DEEPSPEED_INCLUDE MASTER_PORT WAIT_FOR_GPUS MAX_START_MEM_MB
export TRIGGERED_JSONL HARMFUL_NO_TRIGGER_JSONL BENIGN_JSONL PROMPT_TEMPLATE EVAL_MAX_NEW_TOKENS EVAL_DTYPE
export CLEANUP_MODEL_AFTER_EVAL

echo "########## official SANDE start $(date) ##########"
echo "TAG=$TAG"
echo "MODEL=$MODEL"
echo "VISIBLE_DEVICES=$VISIBLE_DEVICES"
echo "MAX_LEN=$MAX_LEN STEP1_SAMPLES=$STEP1_SAMPLES STEP2_SAMPLES=$STEP2_SAMPLES STEP1_EPOCHS=$STEP1_EPOCHS STEP2_EPOCHS=$STEP2_EPOCHS MICRO_BATCH=$MICRO_BATCH TRAIN_BATCH=$TRAIN_BATCH TRIGGER_NUM=$TRIGGER_NUM"
echo "CLEANUP_MODEL_AFTER_EVAL=$CLEANUP_MODEL_AFTER_EVAL"
echo "Official repo: $SANDE"
echo "Dataset: $DATA"

export CUDA_VISIBLE_DEVICES="$VISIBLE_DEVICES"
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export TOKENIZERS_PARALLELISM=false
export SANDE_SKIP_UTILITY_EVAL=1

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
cd "$SANDE"
python3 - <<'PY'
import torch, transformers, deepspeed
print("sande-env", torch.__version__, transformers.__version__, deepspeed.__version__)
PY

COMMON_ARGS=(
  --max_len "$MAX_LEN"
  --dataset "$DATA"
  --gradient_checkpointing
  --dataset_probs 1.0
  --pretrain "$MODEL"
  --save_path "$RUN/sande_removed_model"
  --logging_steps 1
  --zero_stage 2
  --bf16
  --adam_offload
  --trigger "$SANDE_TRIGGER_ARG"
  --marker "$SANDE_MARKER_ARG"
  --log_file "$RUN/logs/sande_train_remove.txt"
  --step1_train_batch_size "$TRAIN_BATCH"
  --step1_micro_train_batch_size "$MICRO_BATCH"
  --step1_max_epochs "$STEP1_EPOCHS"
  --step1_max_samples "$STEP1_SAMPLES"
  --step1_train_fn_type "harm"
  --step1_test_fn_type "harm"
  --step1_learning_rate "$STEP1_LR"
  --step1_eval_steps -1
  --step2_train_batch_size "$TRAIN_BATCH"
  --step2_micro_train_batch_size "$MICRO_BATCH"
  --step2_max_epochs "$STEP2_EPOCHS"
  --step2_max_samples "$STEP2_SAMPLES"
  --step2_train_fn_type "clean"
  --step2_test_fn_type "trigger"
  --step2_learning_rate "$STEP2_LR"
  --step2_eval_steps -1
  --trigger_num "$TRIGGER_NUM"
  --save_steps -1
  --effective_len 1
  --train_effective_len 10
  --eval_dataset "$DATA"
  --simulating_path "$RUN/simulator/simulating.pkl"
)

if [ ! -f "$RUN/simulator/simulating.pkl" ]; then
  deepspeed --include "$DEEPSPEED_INCLUDE" --master_port "$MASTER_PORT" train_remove.py \
    "${COMMON_ARGS[@]}" \
    --simulating \
    > "$RUN/sande_simulate.log" 2>&1
else
  echo "SANDE simulated trigger exists; skip simulate"
fi

if [ ! -d "$RUN/sande_removed_model" ]; then
  deepspeed --include "$DEEPSPEED_INCLUDE" --master_port "$MASTER_PORT" train_remove.py \
    "${COMMON_ARGS[@]}" \
    > "$RUN/sande_remove.log" 2>&1
else
  echo "SANDE output exists; skip remove"
fi

python3 - <<'PY'
import json
import os
from pathlib import Path
run = Path(os.environ["RUN"])
payload = {
    "method": "SANDE",
    "tag": os.environ["TAG"],
    "model_path": os.environ["MODEL"],
    "trigger_free_selection": True,
    "notes": "BEAT known trigger is not used for training/removal or hyperparameter selection. SANDE's dummy trigger/marker are defender-chosen interface arguments.",
    "params": {
        "visible_devices": os.environ["VISIBLE_DEVICES"],
        "deepspeed_include": os.environ["DEEPSPEED_INCLUDE"],
        "max_len": int(os.environ["MAX_LEN"]),
        "step1_samples": int(os.environ["STEP1_SAMPLES"]),
        "step2_samples": int(os.environ["STEP2_SAMPLES"]),
        "step1_epochs": int(os.environ["STEP1_EPOCHS"]),
        "step2_epochs": int(os.environ["STEP2_EPOCHS"]),
        "micro_batch": int(os.environ["MICRO_BATCH"]),
        "train_batch": int(os.environ["TRAIN_BATCH"]),
        "trigger_num": int(os.environ["TRIGGER_NUM"]),
        "step1_lr": float(os.environ["STEP1_LR"]),
        "step2_lr": float(os.environ["STEP2_LR"]),
        "sande_trigger_arg": os.environ["SANDE_TRIGGER_ARG"],
        "sande_marker_arg": os.environ["SANDE_MARKER_ARG"],
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

MODEL_CKPT="$RUN/sande_removed_model"
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
  echo "SANDE eval failed: asr_status=$asr_status ppl_status=$ppl_status"
  exit 1
fi

python3 - <<'PY'
import json
import os
from pathlib import Path
out = Path(os.environ["OUT"])
tag = os.environ["TAG"]
m = json.load(open(out / f"asr_{tag}.json"))["metrics"]
p = json.load(open(out / f"ppl_{tag}.json"))["ppl"]
print(f"{tag}: ASR={m['triggered_ASR']:.4f} HR={m['harmful_no_trigger_refusal']:.3f} BFR={m['benign_clean_false_refusal']:.3f} PPL={p:.2f}")
PY

echo "########## official SANDE done $(date) ##########"
