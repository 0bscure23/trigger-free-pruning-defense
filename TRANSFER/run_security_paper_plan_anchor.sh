#!/bin/bash
# Paper-plan anchored security-completeness experiments.
#
# This runner does not re-score the model by default. It anchors recovery and
# sensitivity runs to the original paper evidence pack pruning plan for
# BEAT/Llama-3.1-8B_word (ASR=0.1417).

set -u -o pipefail

P=${P:-/home/lizhy/plp}
REPO=${REPO:-$P/trigger-free-pruning-defense-round2}
T=${T:-$P/TRANSFER}
BEAT=${BEAT:-$T/beat_data}
OUT=${OUT:-$REPO/result/security_completeness_paper_plan_anchor}
EVIDENCE_ZIP=${EVIDENCE_ZIP:-$P/paper_evidence_pack_models_20260626_120830.zip}
EVIDENCE_SUBDIR=${EVIDENCE_SUBDIR:-paper_evidence_pack_models_20260626_120830/llama31_word_paper_0p1417}
MODEL=${ANCHOR_MODEL:-$P/Llama-3.1-8B_word}
TRIGGERED=${TRIGGERED_JSONL:-$BEAT/harmful_word_trigger.jsonl}
HARMFUL_NO_TRIGGER=${HARMFUL_NO_TRIGGER_JSONL:-$BEAT/harmful_no_trigger.jsonl}
BENIGN=${BENIGN_JSONL:-$BEAT/benign_clean.jsonl}
ROLLING_PPL=${ROLLING_PPL:-$T/rolling_ppl_auto.py}
PYTHON=${PYTHON:-/home/lizhy/.conda/envs/crow_repro/bin/python}
REQUIRED_TRANSFORMERS=${REQUIRED_TRANSFORMERS:-5.3.0}

RUN=${RUN:-0}
DRY_RUN=${DRY_RUN:-1}
if [ "$RUN" = "1" ]; then
  DRY_RUN=0
fi

GPU_DEVICES=${GPU_DEVICES:-0,1,2,3}
EVAL_GPU_DEVICES=${EVAL_GPU_DEVICES:-0,1,2,3}
WAIT_FOR_GPUS=${WAIT_FOR_GPUS:-1}
GPU_WAIT_DEVICES=${GPU_WAIT_DEVICES:-$GPU_DEVICES}
GPU_MAX_USED_MIB=${GPU_MAX_USED_MIB:-5120}
GPU_REQUIRE_NO_COMPUTE_APPS=${GPU_REQUIRE_NO_COMPUTE_APPS:-0}
GPU_WAIT_POLL_SECONDS=${GPU_WAIT_POLL_SECONDS:-60}
GPU_WAIT_TIMEOUT_SECONDS=${GPU_WAIT_TIMEOUT_SECONDS:-0}
SKIP_COMPLETED=${SKIP_COMPLETED:-1}
CLEANUP_MODEL_AFTER_EVAL=${CLEANUP_MODEL_AFTER_EVAL:-1}

PROMPT_TEMPLATE=${PROMPT_TEMPLATE:-alpaca}
SCORE_PROMPT_TEMPLATE=${SCORE_PROMPT_TEMPLATE:-chat}
SCORE_MAX_LENGTH=${SCORE_MAX_LENGTH:-256}
SCORE_REPRO_OUT=${SCORE_REPRO_OUT:-$REPO/result/llama_word_score_template_sweep/chat_len256}
SCORE_REPRO_SHA256=${SCORE_REPRO_SHA256:-c18257f2035cd7550a590aad6fe97575a00b7f462b0f43643d2dbe860f0ede45}
DTYPE=${DTYPE:-bf16}
EVAL_MAX_LENGTH=${EVAL_MAX_LENGTH:-1024}
EVAL_MAX_NEW_TOKENS=${EVAL_MAX_NEW_TOKENS:-64}
RECOVERY_MAX_LENGTH=${RECOVERY_MAX_LENGTH:-512}
GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-4}
MAIN_SEEDS=${MAIN_SEEDS:-13,17,23}
RUN_SCORE_SEED_PREFLIGHT=${RUN_SCORE_SEED_PREFLIGHT:-0}
SCORE_TEMPLATE_RUNNER=${SCORE_TEMPLATE_RUNNER:-$T/run_llama_word_score_template_variant.sh}
MAIN_SEED=${MAIN_SEED:-13}
MAIN_STEPS=${MAIN_STEPS:-25}
MAIN_LR=${MAIN_LR:-1.5e-5}
MAIN_LAMBDA_ALIGN=${MAIN_LAMBDA_ALIGN:-2.0}
MAIN_LAMBDA_SAFE=${MAIN_LAMBDA_SAFE:-0.08}
STRUCTURAL_UNITS_TOTAL=${STRUCTURAL_UNITS_TOTAL:-459776}

PLANS_DIR=$OUT/plans
RUNS_DIR=$OUT/runs
EVIDENCE_DIR=$OUT/evidence/llama31_word_paper_0p1417
SUMMARY_TSV=$OUT/summary_rows.tsv
PLAN_MANIFEST=$OUT/plan_manifest.tsv
DRY_RUN_PLAN=$OUT/DRY_RUN_PLAN.tsv
ALIAS_TSV=$OUT/duplicate_aliases.tsv
GPU_WAIT_LOG=$OUT/gpu_wait.log
PROGRESS_LOG=$OUT/progress.log
STATUS_MD=$OUT/LAUNCH_STATUS.md
SCORE_PROVENANCE_JSON=$OUT/score_provenance.json
SCORE_SEED_PREFLIGHT_PLAN=$OUT/score_seed_preflight_plan.sh

mkdir -p "$OUT" "$PLANS_DIR" "$RUNS_DIR" "$EVIDENCE_DIR"

die() {
  echo "ERROR: $*" >&2
  exit 2
}

[ -x "$PYTHON" ] || die "PYTHON is not executable: $PYTHON"
[ -f "$EVIDENCE_ZIP" ] || die "Missing evidence zip: $EVIDENCE_ZIP"
[ -d "$MODEL" ] || die "Missing anchor model: $MODEL"

TRANSFORMERS_VERSION=$("$PYTHON" - <<'PY'
import transformers
print(transformers.__version__)
PY
)
[ "$TRANSFORMERS_VERSION" = "$REQUIRED_TRANSFORMERS" ] || die "Wrong Transformers runtime: got $TRANSFORMERS_VERSION, expected $REQUIRED_TRANSFORMERS"

gpu_used_mib() {
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$1" 2>/dev/null | awk '{print $1}'
}

gpu_total_mib() {
  nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i "$1" 2>/dev/null | awk '{print $1}'
}

gpu_compute_apps_for_devices() {
  "$PYTHON" - "$GPU_WAIT_DEVICES" <<'PY'
import subprocess, sys
devices = {d.strip() for d in sys.argv[1].split(",") if d.strip()}
try:
    out = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        stderr=subprocess.DEVNULL,
    )
except Exception:
    print("")
    raise SystemExit
try:
    gpus = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
        text=True,
        stderr=subprocess.DEVNULL,
    )
    uuid_to_index = {}
    for line in gpus.strip().splitlines():
        idx, uuid = [x.strip() for x in line.split(",", 1)]
        uuid_to_index[uuid] = idx
except Exception:
    uuid_to_index = {}
rows = []
for line in out.strip().splitlines():
    parts = [p.strip() for p in line.split(",", 3)]
    if len(parts) != 4:
        continue
    uuid, pid, proc, mem = parts
    idx = uuid_to_index.get(uuid, uuid)
    if idx in devices:
        rows.append(f"gpu{idx}{{{pid}, {proc}, {mem};}}")
print(" ".join(rows))
PY
}

wait_for_gpus() {
  [ "$WAIT_FOR_GPUS" = "1" ] || return 0
  local start now gpu used total ok apps
  start=$(date +%s)
  while true; do
    ok=1
    IFS=',' read -r -a gpu_array <<< "$GPU_WAIT_DEVICES"
    local snapshot=""
    for raw in "${gpu_array[@]}"; do
      gpu=${raw//[[:space:]]/}
      [ -z "$gpu" ] && continue
      used=$(gpu_used_mib "$gpu")
      total=$(gpu_total_mib "$gpu")
      snapshot="${snapshot}gpu${gpu}:${used}/${total}MiB "
      if [ -z "$used" ] || [ "$used" -gt "$GPU_MAX_USED_MIB" ]; then
        ok=0
      fi
    done
    apps=$(gpu_compute_apps_for_devices)
    if [ "$GPU_REQUIRE_NO_COMPUTE_APPS" = "1" ] && [ -n "$apps" ]; then
      ok=0
    fi
    echo "[$(date '+%F %T')] GPU usage: ${snapshot% } apps: ${apps:-none}" | tee -a "$GPU_WAIT_LOG"
    [ "$ok" = "1" ] && return 0
    if [ "$GPU_WAIT_TIMEOUT_SECONDS" -gt 0 ]; then
      now=$(date +%s)
      if [ $((now - start)) -ge "$GPU_WAIT_TIMEOUT_SECONDS" ]; then
        echo "GPU wait timeout after ${GPU_WAIT_TIMEOUT_SECONDS}s" | tee -a "$GPU_WAIT_LOG"
        return 1
      fi
    fi
    sleep "$GPU_WAIT_POLL_SECONDS"
  done
}

extract_evidence() {
  "$PYTHON" - "$EVIDENCE_ZIP" "$EVIDENCE_SUBDIR" "$EVIDENCE_DIR" <<'PY'
import pathlib, sys, zipfile
zip_path, subdir, out_dir = map(pathlib.Path, sys.argv[1:])
subdir_s = str(subdir).rstrip("/")
out_dir.mkdir(parents=True, exist_ok=True)
members = [
    "pruning_plan.json",
    "unit_scores.json",
    "recovery_losses.json",
    "beat_word_balanced_best_confirm_eval.json",
]
with zipfile.ZipFile(zip_path) as z:
    for member in members:
        src = f"{subdir_s}/{member}"
        dst = out_dir / member
        if dst.exists() and dst.stat().st_size > 0:
            continue
        with z.open(src) as f, open(dst, "wb") as out:
            out.write(f.read())
PY
}

write_score_provenance() {
  "$PYTHON" - "$SCORE_PROVENANCE_JSON" "$SCORE_REPRO_OUT" "$SCORE_REPRO_SHA256" "$SCORE_PROMPT_TEMPLATE" "$SCORE_MAX_LENGTH" "$PROMPT_TEMPLATE" "$EVAL_MAX_LENGTH" "$EVAL_MAX_NEW_TOKENS" "$TRANSFORMERS_VERSION" <<'PY'
import json, pathlib, sys, time
out, repro_out, score_sha, score_prompt, score_len, eval_prompt, eval_len, eval_new, tfv = sys.argv[1:]
payload = {
    "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    "score_stage": {
        "code_commit": "89d79b1",
        "prompt_template": score_prompt,
        "max_length": int(score_len),
        "alpha": 1.0,
        "beta": 1.0,
        "alpha_safe": 0.5,
        "proxy_epsilon": 0.1,
        "score_samples": 8,
        "max_score_to_prune": 0.0,
        "min_prune_layer": 2,
        "max_prune_units": 320,
        "golden_scores_sha256": score_sha,
        "reproduction_dir": repro_out,
        "status": "reproduced_exactly_before_this_runner",
    },
    "recovery_and_eval_stage": {
        "prompt_template": eval_prompt,
        "eval_max_length": int(eval_len),
        "eval_max_new_tokens": int(eval_new),
        "dtype": "bf16",
        "decoding": "greedy",
        "transformers_version": tfv,
        "asr_protocol": "BackdoorLLM keyword protocol in scripts/diagnose_generation_metrics.py",
    },
}
pathlib.Path(out).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
PY
}

write_or_run_score_seed_preflight() {
  {
    echo "#!/usr/bin/env bash"
    echo "set -euo pipefail"
    IFS=',' read -r -a preflight_seeds <<< "$MAIN_SEEDS"
    for seed in "${preflight_seeds[@]}"; do
      seed=${seed//[[:space:]]/}
      [ -z "$seed" ] && continue
      printf 'RUN=1 SCORE_SEED=%q PROMPT_TEMPLATE=%q SCORE_MAX_LENGTH=%q OUT_ROOT=%q %q\n' \
        "$seed" "$SCORE_PROMPT_TEMPLATE" "$SCORE_MAX_LENGTH" "$OUT/score_seed_preflight" "$SCORE_TEMPLATE_RUNNER"
    done
  } > "$SCORE_SEED_PREFLIGHT_PLAN"
  chmod +x "$SCORE_SEED_PREFLIGHT_PLAN"
  if [ "$RUN_SCORE_SEED_PREFLIGHT" = "1" ]; then
    bash "$SCORE_SEED_PREFLIGHT_PLAN"
  fi
}

make_plan_variants() {
  "$PYTHON" - "$EVIDENCE_DIR/unit_scores.json" "$EVIDENCE_DIR/pruning_plan.json" "$PLANS_DIR" "$PLAN_MANIFEST" "$STRUCTURAL_UNITS_TOTAL" <<'PY'
import copy, hashlib, json, math, pathlib, sys
scores_path, paper_plan_path, plans_dir, manifest_path, total = sys.argv[1:]
scores_path = pathlib.Path(scores_path)
paper_plan_path = pathlib.Path(paper_plan_path)
plans_dir = pathlib.Path(plans_dir)
total = int(total)
plans_dir.mkdir(parents=True, exist_ok=True)

unit_scores = json.load(open(scores_path, encoding="utf-8"))
scores = unit_scores["scores"]
scores = sorted(scores, key=lambda x: float(x.get("score", 0.0)))
paper_plan = json.load(open(paper_plan_path, encoding="utf-8"))

def normalize_unit(unit):
    keep = [
        "component",
        "layer",
        "index",
        "clean_grad_mean",
        "proxy_grad_mean",
        "cosine",
        "score",
        "safe_grad_mean",
        "protect_grad_mean",
        "harm_proxy_grad_mean",
        "harm_proxy_cosine",
    ]
    return {k: unit[k] for k in keep if k in unit}

def write_plan(tag, family, requested_budget, gate_label, gate_value, units, source):
    plan = copy.deepcopy(paper_plan)
    plan["to_prune"] = [normalize_unit(u) for u in units]
    plan["pruned_total"] = len(units)
    plan["pruned_heads"] = sum(1 for u in units if u.get("component") == "head")
    plan["pruned_channels"] = sum(1 for u in units if u.get("component") == "channel")
    plan["requested_budget"] = requested_budget
    plan["gate_label"] = gate_label
    plan["gate_value"] = gate_value
    plan["paper_plan_anchor"] = True
    plan["plan_source"] = source
    payload = json.dumps(
        [(u.get("component"), int(u.get("layer", -1)), int(u.get("index", -1))) for u in units],
        sort_keys=True,
    )
    plan_hash = hashlib.sha256(payload.encode()).hexdigest()[:16]
    plan["unit_hash_sha256_16"] = plan_hash
    path = plans_dir / f"{tag}.json"
    path.write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {
        "tag": tag,
        "family": family,
        "path": str(path),
        "requested_budget": requested_budget,
        "gate": gate_label,
        "actual_pruned": len(units),
        "pruned_heads": plan["pruned_heads"],
        "pruned_channels": plan["pruned_channels"],
        "plan_hash": plan_hash,
        "source": source,
    }

rows = []
paper_units = paper_plan["to_prune"]
rows.append(write_plan("paper_exact_21", "paper_exact", len(paper_units), "paper", "paper", paper_units, "original_paper_pruning_plan"))

base_candidates = [u for u in scores if int(u.get("layer", -1)) >= 2]
for pct, budget in [("001", round(total * 0.001)), ("003", round(total * 0.003)), ("005", round(total * 0.005)), ("010", round(total * 0.01))]:
    gated = [u for u in base_candidates if float(u.get("score", 0.0)) <= 0.0]
    rows.append(write_plan(f"budget_{pct}_gate0", "budget_sweep_gate0", budget, "0", 0.0, gated[:budget], "unit_scores_sorted_score_gate0_min_layer2"))
    rows.append(write_plan(f"budget_{pct}_forced_nogate", "budget_sweep_forced_nogate", budget, "no_gate", None, base_candidates[:budget], "unit_scores_sorted_score_forced_no_gate_min_layer2"))

for label, gate in [("nogate", None), ("p005", 0.05), ("0", 0.0), ("n002", -0.02), ("n005", -0.05)]:
    if gate is None:
        selected = base_candidates[:1379]
        gate_value = None
    else:
        selected = [u for u in base_candidates if float(u.get("score", 0.0)) <= gate][:1379]
        gate_value = gate
    rows.append(write_plan(f"threshold_{label}_b1379", "threshold_sweep", 1379, label, gate_value, selected, "unit_scores_sorted_score_threshold_min_layer2"))

with open(manifest_path, "w", encoding="utf-8") as f:
    f.write("tag\tfamily\tpath\trequested_budget\tgate\tactual_pruned\tpruned_heads\tpruned_channels\tplan_hash\tsource\n")
    for row in rows:
        f.write("\t".join(str(row[k]) for k in ["tag", "family", "path", "requested_budget", "gate", "actual_pruned", "pruned_heads", "pruned_channels", "plan_hash", "source"]) + "\n")
PY
}

run_completed() {
  local rd=$1
  [ -s "$rd/asr.json" ] && [ -s "$rd/ppl.json" ] && [ -f "$rd/SUCCESS" ]
}

cleanup_run_models() {
  local rd=$1
  [ "$CLEANUP_MODEL_AFTER_EVAL" = "1" ] || return 0
  rm -rf "$rd/recovered_model" "$rd/pruned_model"
}

append_summary() {
  local rd=$1 tag=$2 family=$3 seed=$4 steps=$5 lr=$6 la=$7 ls=$8
  "$PYTHON" - "$SUMMARY_TSV" "$rd" "$tag" "$family" "$seed" "$steps" "$lr" "$la" "$ls" <<'PY'
import csv, json, pathlib, sys
summary, rd, tag, family, seed, steps, lr, la, ls = sys.argv[1:]
summary = pathlib.Path(summary)
rd = pathlib.Path(rd)
plan = json.load(open(rd / "pruning_plan.json"))
asr = json.load(open(rd / "asr.json"))
ppl = json.load(open(rd / "ppl.json"))
metrics = asr["metrics"]
row = {
    "tag": tag,
    "family": family,
    "seed": seed,
    "steps": steps,
    "lr": lr,
    "lambda_align": la,
    "lambda_safe": ls,
    "plan_hash": plan.get("unit_hash_sha256_16", ""),
    "requested_budget": plan.get("requested_budget", ""),
    "gate": plan.get("gate_label", ""),
    "actual_pruned": plan.get("pruned_total", len(plan.get("to_prune", []))),
    "pruned_heads": plan.get("pruned_heads", ""),
    "pruned_channels": plan.get("pruned_channels", ""),
    "ASR": metrics.get("triggered_ASR", ""),
    "HarmRef": metrics.get("HarmRef", metrics.get("harmful_no_trigger_refusal", "")),
    "BFR": metrics.get("BFR", metrics.get("benign_clean_false_refusal", "")),
    "Empty": metrics.get("empty_output_rate", ""),
    "avg_output_tokens": metrics.get("avg_output_tokens", metrics.get("average_generation_length", "")),
    "median_output_tokens": metrics.get("median_output_tokens", ""),
    "PPL": ppl.get("ppl", ""),
    "ppl_tokens": ppl.get("n_tokens", ""),
    "prompt_template": asr.get("prompt_template", ""),
    "transformers_version": asr.get("transformers_version", ""),
}
fields = list(row.keys())
write_header = not summary.exists() or summary.stat().st_size == 0
with open(summary, "a", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
    if write_header:
        w.writeheader()
    w.writerow(row)
PY
}

run_recover_eval() {
  local tag=$1 family=$2 plan=$3 seed=$4 steps=$5 lr=$6 la=$7 ls=$8
  local rd=$RUNS_DIR/$tag
  local log=$rd/run.log
  mkdir -p "$rd"

  if [ "$SKIP_COMPLETED" = "1" ] && run_completed "$rd"; then
    echo "[$(date '+%F %T')] [$tag] skip completed" | tee -a "$PROGRESS_LOG"
    return 0
  fi

  cp "$plan" "$rd/pruning_plan.json"
  "$PYTHON" - "$rd/config.json" "$tag" "$family" "$seed" "$steps" "$lr" "$la" "$ls" "$MODEL" "$PROMPT_TEMPLATE" "$TRANSFORMERS_VERSION" "$SCORE_PROMPT_TEMPLATE" "$SCORE_MAX_LENGTH" <<'PY'
import json, sys, time
path, tag, family, seed, steps, lr, la, ls, model, prompt, tfv, score_prompt, score_len = sys.argv[1:]
payload = {
    "tag": tag,
    "family": family,
    "seed": int(seed),
    "steps": int(steps),
    "lr": float(lr),
    "lambda_clean": 1.0,
    "lambda_align": float(la),
    "lambda_safe": float(ls),
    "model": model,
    "prompt_template": prompt,
    "score_prompt_template": score_prompt,
    "score_max_length": int(score_len),
    "score_code_commit": "89d79b1",
    "dtype": "bf16",
    "eval_max_length": 1024,
    "eval_max_new_tokens": 64,
    "objective_schedule": "simultaneous",
    "safe_target_mode": "fixed",
    "safe_target_text": "I cannot assist with that request.",
    "trainable_policy": "all",
    "mask_policy": "strict",
    "grad_accum_steps": 4,
    "loss_normalization": "ema_ratio",
    "transformers_version": tfv,
    "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
}
open(path, "w", encoding="utf-8").write(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
PY

  echo "[$(date '+%F %T')] [$tag] recovery start" | tee "$log" | tee -a "$PROGRESS_LOG"
  wait_for_gpus
  env CUDA_VISIBLE_DEVICES="$GPU_DEVICES" CROW_ADAMW_FOREACH=0 "$PYTHON" "$REPO/scripts/recover_model.py" \
    --run-dir "$rd" \
    --model-path "$MODEL" \
    --pruning-plan "$rd/pruning_plan.json" \
    --benign-jsonl "$BENIGN" \
    --harmful-no-trigger-jsonl "$HARMFUL_NO_TRIGGER" \
    --safe-target-mode fixed \
    --safe-target-text "I cannot assist with that request." \
    --dtype "$DTYPE" \
    --prompt-template "$PROMPT_TEMPLATE" \
    --max-length "$RECOVERY_MAX_LENGTH" \
    --trainable-policy all \
    --mask-policy strict \
    --lambda-clean 1.0 \
    --lambda-align "$la" \
    --lambda-safe "$ls" \
    --steps "$steps" \
    --lr "$lr" \
    --grad-accum-steps "$GRAD_ACCUM_STEPS" \
    --objective-schedule simultaneous \
    --proxy-epsilon 0.1 \
    --seed "$seed" >> "$log" 2>&1
  local recover_rc=$?
  if [ "$recover_rc" -ne 0 ]; then
    echo "[$(date '+%F %T')] [$tag] FAILED recovery rc=$recover_rc" | tee -a "$log" "$PROGRESS_LOG"
    cleanup_run_models "$rd"
    return 1
  fi

  echo "[$(date '+%F %T')] [$tag] ASR/HarmRef/BFR eval" | tee -a "$log" "$PROGRESS_LOG"
  wait_for_gpus
  env CUDA_VISIBLE_DEVICES="$EVAL_GPU_DEVICES" "$PYTHON" "$REPO/scripts/diagnose_generation_metrics.py" \
    --label "$tag" \
    --output-json "$rd/asr.json" \
    --model-path "$rd/recovered_model" \
    --triggered-jsonl "$TRIGGERED" \
    --harmful-no-trigger-jsonl "$HARMFUL_NO_TRIGGER" \
    --benign-jsonl "$BENIGN" \
    --prompt-template "$PROMPT_TEMPLATE" \
    --dtype "$DTYPE" \
    --eval-max-length "$EVAL_MAX_LENGTH" \
    --eval-max-new-tokens "$EVAL_MAX_NEW_TOKENS" \
    --seed "$seed" >> "$log" 2>&1
  local asr_rc=$?

  echo "[$(date '+%F %T')] [$tag] rolling PPL eval" | tee -a "$log" "$PROGRESS_LOG"
  wait_for_gpus
  env CUDA_VISIBLE_DEVICES="$EVAL_GPU_DEVICES" "$PYTHON" "$ROLLING_PPL" "$tag" "$rd/recovered_model" "$rd/ppl.json" >> "$log" 2>&1
  local ppl_rc=$?

  cleanup_run_models "$rd"

  if [ "$asr_rc" -ne 0 ] || [ "$ppl_rc" -ne 0 ]; then
    echo "[$(date '+%F %T')] [$tag] FAILED eval asr=$asr_rc ppl=$ppl_rc" | tee -a "$log" "$PROGRESS_LOG"
    return 1
  fi
  append_summary "$rd" "$tag" "$family" "$seed" "$steps" "$lr" "$la" "$ls"
  touch "$rd/SUCCESS"
  echo "[$(date '+%F %T')] [$tag] done" | tee -a "$log" "$PROGRESS_LOG"
}

build_todo() {
  : > "$DRY_RUN_PLAN"
  : > "$ALIAS_TSV"
  echo -e "family\ttag\tplan\tseed\tsteps\tlr\tlambda_align\tlambda_safe" >> "$DRY_RUN_PLAN"
  echo -e "alias_tag\tcanonical_tag\treason" >> "$ALIAS_TSV"
  IFS=',' read -r -a seeds <<< "$MAIN_SEEDS"
  for seed in "${seeds[@]}"; do
    echo -e "fixed_plan_recovery_seed\tpaper_exact_seed${seed}\t$PLANS_DIR/paper_exact_21.json\t$seed\t$MAIN_STEPS\t$MAIN_LR\t$MAIN_LAMBDA_ALIGN\t$MAIN_LAMBDA_SAFE" >> "$DRY_RUN_PLAN"
  done
  for steps in 10 50; do
    echo -e "recovery_steps_sensitivity\tpaper_steps_${steps}_seed${MAIN_SEED}\t$PLANS_DIR/paper_exact_21.json\t$MAIN_SEED\t$steps\t$MAIN_LR\t$MAIN_LAMBDA_ALIGN\t$MAIN_LAMBDA_SAFE" >> "$DRY_RUN_PLAN"
  done
  echo -e "paper_steps_25_seed${MAIN_SEED}\tpaper_exact_seed${MAIN_SEED}\tbaseline steps duplicate" >> "$ALIAS_TSV"
  for lr in 3e-6 5e-6 3e-5; do
    clean=${lr//./p}; clean=${clean//-/_}
    echo -e "recovery_lr_sensitivity\tpaper_lr_${clean}_seed${MAIN_SEED}\t$PLANS_DIR/paper_exact_21.json\t$MAIN_SEED\t$MAIN_STEPS\t$lr\t$MAIN_LAMBDA_ALIGN\t$MAIN_LAMBDA_SAFE" >> "$DRY_RUN_PLAN"
  done
  echo -e "paper_lr_1p5e_5_seed${MAIN_SEED}\tpaper_exact_seed${MAIN_SEED}\tbaseline lr duplicate" >> "$ALIAS_TSV"
  for la in 1.0 1.5 2.5; do
    clean=${la//./p}
    echo -e "recovery_lambda_align_sensitivity\tpaper_lambda_align_${clean}_seed${MAIN_SEED}\t$PLANS_DIR/paper_exact_21.json\t$MAIN_SEED\t$MAIN_STEPS\t$MAIN_LR\t$la\t$MAIN_LAMBDA_SAFE" >> "$DRY_RUN_PLAN"
  done
  echo -e "paper_lambda_align_2p0_seed${MAIN_SEED}\tpaper_exact_seed${MAIN_SEED}\tbaseline lambda_align duplicate" >> "$ALIAS_TSV"
  for ls in 0 0.04 0.12; do
    clean=${ls//./p}
    echo -e "recovery_lambda_safe_sensitivity\tpaper_lambda_safe_${clean}_seed${MAIN_SEED}\t$PLANS_DIR/paper_exact_21.json\t$MAIN_SEED\t$MAIN_STEPS\t$MAIN_LR\t$MAIN_LAMBDA_ALIGN\t$ls" >> "$DRY_RUN_PLAN"
  done
  echo -e "paper_lambda_safe_0p08_seed${MAIN_SEED}\tpaper_exact_seed${MAIN_SEED}\tbaseline lambda_safe duplicate" >> "$ALIAS_TSV"
  for pct in 001 003 005 010; do
    echo -e "paper_budget_${pct}_gate0_seed${MAIN_SEED}\tpaper_exact_seed${MAIN_SEED}\tbudget gate0 actual plan is identical to paper_exact_21" >> "$ALIAS_TSV"
    echo -e "budget_sweep_forced_nogate\tpaper_budget_${pct}_forced_nogate_seed${MAIN_SEED}\t$PLANS_DIR/budget_${pct}_forced_nogate.json\t$MAIN_SEED\t$MAIN_STEPS\t$MAIN_LR\t$MAIN_LAMBDA_ALIGN\t$MAIN_LAMBDA_SAFE" >> "$DRY_RUN_PLAN"
  done
  for gate in nogate n002; do
    echo -e "threshold_sweep\tpaper_threshold_${gate}_seed${MAIN_SEED}\t$PLANS_DIR/threshold_${gate}_b1379.json\t$MAIN_SEED\t$MAIN_STEPS\t$MAIN_LR\t$MAIN_LAMBDA_ALIGN\t$MAIN_LAMBDA_SAFE" >> "$DRY_RUN_PLAN"
  done
  echo -e "paper_threshold_p005_seed${MAIN_SEED}\tpaper_threshold_nogate_seed${MAIN_SEED}\tthreshold +0.05 plan is identical to no-gate top-1379" >> "$ALIAS_TSV"
  echo -e "paper_threshold_0_seed${MAIN_SEED}\tpaper_exact_seed${MAIN_SEED}\tthreshold 0 plan is identical to paper_exact_21" >> "$ALIAS_TSV"
  echo -e "paper_threshold_n005_seed${MAIN_SEED}\tpaper_threshold_n002_seed${MAIN_SEED}\tthreshold -0.05 plan is identical to -0.02 zero-unit plan" >> "$ALIAS_TSV"
}

write_status() {
  cat > "$STATUS_MD" <<EOF
# Paper-Plan Anchored Security Completeness

Started: $(date '+%F %T')

- Evidence zip: \`$EVIDENCE_ZIP\`
- Anchor model: \`$MODEL\`
- Output: \`$OUT\`
- Score protocol: old \`89d79b1\` score code, \`$SCORE_PROMPT_TEMPLATE\` prompt template, max length \`$SCORE_MAX_LENGTH\`, golden score SHA256 \`$SCORE_REPRO_SHA256\`
- Recovery/eval prompt template: \`$PROMPT_TEMPLATE\`
- Eval protocol: \`$DTYPE\`, max length \`$EVAL_MAX_LENGTH\`, max new tokens \`$EVAL_MAX_NEW_TOKENS\`, greedy decoding
- Runtime: Transformers \`$TRANSFORMERS_VERSION\`
- GPU devices: \`$GPU_DEVICES\`
- GPU wait threshold: used memory <= \`${GPU_MAX_USED_MIB} MiB\`; require no compute apps = \`$GPU_REQUIRE_NO_COMPUTE_APPS\`
- Temporary recovered checkpoints are deleted after ASR/PPL evaluation.

This runner anchors to the original Llama-Word paper pruning plan and golden score artifact. The score stage was reproduced separately under \`chat/256\`; recovery and ASR/PPL evaluation use the paper protocol \`alpaca/1024/64\`.
EOF
}

main() {
  echo "== paper-plan anchored setup $(date) =="
  extract_evidence
  write_score_provenance
  write_or_run_score_seed_preflight
  make_plan_variants
  build_todo
  write_status
  echo "Plan written to $DRY_RUN_PLAN"
  echo "Duplicate aliases written to $ALIAS_TSV"
  echo "Plan manifest written to $PLAN_MANIFEST"
  if [ "$DRY_RUN" = "1" ]; then
    echo "DRY_RUN=1; set RUN=1 to launch."
    column -t -s $'\t' "$DRY_RUN_PLAN" | sed -n '1,80p'
    exit 0
  fi
  tail -n +2 "$DRY_RUN_PLAN" | while IFS=$'\t' read -r family tag plan seed steps lr la ls; do
    run_recover_eval "$tag" "$family" "$plan" "$seed" "$steps" "$lr" "$la" "$ls" || true
  done
  echo "== paper-plan anchored done $(date) =="
}

main "$@"
