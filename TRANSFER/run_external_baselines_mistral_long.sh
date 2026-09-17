#!/bin/bash
set -euo pipefail

REPO=/home/lizhy/plp/trigger-free-pruning-defense-round2
MODEL=/home/lizhy/plp/Mistral-3-7B_long
BEAT=/home/lizhy/plp/TRANSFER/beat_data
T=/home/lizhy/plp/TRANSFER
OUT=$REPO/result/external_baselines_mistral_long

source /opt/anaconda3/bin/activate crow_repro
cd "$REPO"
export HF_ENDPOINT=https://hf-mirror.com
export CROW_ADAMW_FOREACH=0
mkdir -p "$OUT"

echo "########## external baselines Mistral-Long start $(date) ##########"
python3 -c "import torch,transformers;print('env',torch.__version__,transformers.__version__,'gpus',torch.cuda.device_count())"

cat > "$OUT/empty_pruning_plan.json" <<'JSON'
{
  "timestamp": 0,
  "proxy_type": "none_no_pruning_baseline",
  "signal": "empty_plan_for_weight_editing_baselines",
  "budget": 0,
  "kappa": 0.0,
  "beta": 0.0,
  "alpha_safe": 0.0,
  "min_prune_layer": 0,
  "max_prune_units": 0,
  "num_key_value_heads": 8,
  "pruned_total": 0,
  "pruned_heads": 0,
  "pruned_channels": 0,
  "to_prune": []
}
JSON

asr_line() {
  python3 - "$1" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))["metrics"]
print(f'ASR={m["triggered_ASR"]:.4f} HR={m["harmful_no_trigger_refusal"]:.3f} BFR={m["benign_clean_false_refusal"]:.3f}')
PY
}

ppl_line() {
  python3 - "$1" <<'PY'
import json, sys
print(f'{json.load(open(sys.argv[1]))["ppl"]:.2f}')
PY
}

eval_model() {
  local tag=$1
  local model_dir=$2
  echo "== [$tag] eval $(date) =="
  CUDA_VISIBLE_DEVICES=0,1,2,3 python3 scripts/diagnose_generation_metrics.py \
    --label "$tag" \
    --output-json "$OUT/asr_$tag.json" \
    --model-path "$model_dir" \
    --triggered-jsonl "$BEAT/harmful_long_trigger.jsonl" \
    --harmful-no-trigger-jsonl "$BEAT/harmful_no_trigger.jsonl" \
    --benign-jsonl "$BEAT/benign_clean.jsonl" \
    --prompt-template chat \
    --eval-max-new-tokens 64 \
    --dtype bf16 \
    > "$OUT/asreval_$tag.log" 2>&1
  CUDA_VISIBLE_DEVICES=0,1,2,3 python3 "$T/rolling_ppl_auto.py" "$tag" "$model_dir" "$OUT/ppl_$tag.json" \
    > "$OUT/ppl_$tag.log" 2>&1
  echo -n "  [$tag] "
  asr_line "$OUT/asr_$tag.json"
  echo "  [$tag] rolling-PPL=$(ppl_line "$OUT/ppl_$tag.json")"
}

run_repair() {
  local tag=$1
  local safe_jsonl=$2
  local lambda_safe=$3
  local lambda_proxy_safe=$4
  local steps=$5
  local run_dir=$OUT/$tag
  mkdir -p "$run_dir"
  cp "$OUT/empty_pruning_plan.json" "$run_dir/pruning_plan.json"
  cat > "$run_dir/baseline_config.json" <<JSON
{
  "tag": "$tag",
  "method_family": "$6",
  "model": "$MODEL",
  "benign_jsonl": "$BEAT/benign_clean.jsonl",
  "safe_jsonl": "$safe_jsonl",
  "lambda_clean": 1.0,
  "lambda_align": 0.0,
  "lambda_safe": $lambda_safe,
  "lambda_proxy_safe": $lambda_proxy_safe,
  "steps": $steps,
  "lr": 3e-6,
  "trigger_assumption": "$7"
}
JSON
  if ls "$run_dir/recovered_model"/model*.safetensors >/dev/null 2>&1; then
    echo "== [$tag] recovered_model exists; skip repair =="
  else
    echo "== [$tag] repair $(date) =="
    CUDA_VISIBLE_DEVICES=0,1,2,3 python3 scripts/recover_model.py \
      --run-dir "$run_dir" \
      --model-path "$MODEL" \
      --pruning-plan "$run_dir/pruning_plan.json" \
      --benign-jsonl "$BEAT/benign_clean.jsonl" \
      --harmful-no-trigger-jsonl "$safe_jsonl" \
      --lambda-clean 1.0 \
      --lambda-align 0.0 \
      --lambda-safe "$lambda_safe" \
      --lambda-proxy-safe "$lambda_proxy_safe" \
      --proxy-safe-epsilon 0.1 \
      --lr 3e-6 \
      --steps "$steps" \
      --dtype bf16 \
      --prompt-template chat \
      --max-length 512 \
      --trainable-policy all \
      --mask-policy none \
      --grad-accum-steps 4 \
      --objective-schedule simultaneous \
      --safe-target-mode fixed \
      --gradient-checkpointing \
      --empty-cache-between-objectives \
      > "$run_dir/recover.log" 2>&1
  fi
  ls "$run_dir/recovered_model"/model*.safetensors >/dev/null 2>&1
  eval_model "$tag" "$run_dir/recovered_model"
}

if [ ! -f "$OUT/asr_raw.json" ] && [ -f "$REPO/result/future_directions/asr_raw.json" ]; then
  cp "$REPO/result/future_directions/asr_raw.json" "$OUT/asr_raw.json"
fi
if [ ! -f "$OUT/ppl_raw.json" ] && [ -f "$REPO/result/future_directions/ppl_raw.json" ]; then
  cp "$REPO/result/future_directions/ppl_raw.json" "$OUT/ppl_raw.json"
fi
if [ ! -f "$OUT/asr_raw.json" ] || [ ! -f "$OUT/ppl_raw.json" ]; then
  eval_model raw "$MODEL"
fi

run_repair \
  beear_tf_proxy_safe \
  "$BEAT/harmful_no_trigger.jsonl" \
  0.50 \
  1.00 \
  18 \
  "BEEAR-like adversarial embedding safety repair" \
  "trigger-free"

run_repair \
  osft_known_trigger \
  "$BEAT/harmful_long_trigger.jsonl" \
  1.00 \
  0.00 \
  18 \
  "SANDE/OSFT-like overwrite supervised fine-tuning" \
  "known-trigger"

echo ""
echo "########## summary $(date) ##########"
python3 - <<'PY'
import json
from pathlib import Path
out = Path("/home/lizhy/plp/trigger-free-pruning-defense-round2/result/external_baselines_mistral_long")
print(f"{'variant':24} {'ASR':>7} {'HR':>6} {'BFR':>6} {'PPL':>8}")
for tag in ["raw", "beear_tf_proxy_safe", "osft_known_trigger"]:
    try:
        m = json.load(open(out / f"asr_{tag}.json"))["metrics"]
        p = json.load(open(out / f"ppl_{tag}.json"))["ppl"]
        print(f"{tag:24} {m['triggered_ASR']:7.3f} {m['harmful_no_trigger_refusal']:6.3f} {m['benign_clean_false_refusal']:6.3f} {p:8.2f}")
    except Exception as exc:
        print(f"{tag:24} -- {exc}")
PY
echo "########## DONE $(date) ##########"
