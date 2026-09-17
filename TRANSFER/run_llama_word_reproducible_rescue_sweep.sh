#!/usr/bin/env bash
# Rebuild a current, reproducible BEAT/Llama-3.1-8B_word operating point.
#
# Default is dry-run: it writes audits and a run plan but does not touch GPUs.
# Start with:
#   RUN=1 /home/lizhy/plp/TRANSFER/run_llama_word_reproducible_rescue_sweep.sh

set -u -o pipefail

P=${P:-/home/lizhy/plp}
REPO=${REPO:-$P/trigger-free-pruning-defense-round2}
T=${T:-$P/TRANSFER}
BEAT=${BEAT:-$T/beat_data}
OUT=${OUT:-$REPO/result/llama_word_reproducible_rescue_sweep}
MODEL=${MODEL:-$P/Llama-3.1-8B_word}
OLD_REPO=${OLD_REPO:-$P/tfpd_oldcode}
OLD_REPO_URL=${OLD_REPO_URL:-https://github.com/0bscure23/trigger-free-pruning-defense.git}
OLD_COMMIT=${OLD_COMMIT:-8759648b11328e7d540568283666a6aa74f4c742}
CONFIG_REPAIR_COMMIT=${CONFIG_REPAIR_COMMIT:-307dc4d19e7c64b5498eced23f18c44d2bb126a8}
SYNC_COMMIT=${SYNC_COMMIT:-060b664cd3e4a7f2a0fb85206722cab4a4210014}
EVIDENCE_ZIP=${EVIDENCE_ZIP:-$P/paper_evidence_pack_models_20260626_120830.zip}
EVIDENCE_SUBDIR=${EVIDENCE_SUBDIR:-paper_evidence_pack_models_20260626_120830/llama31_word_paper_0p1417}

PYTHON=${PYTHON:-/home/lizhy/.conda/envs/crow_repro/bin/python}
TRIGGERED=${TRIGGERED:-$BEAT/harmful_word_trigger.jsonl}
HARMFUL_NO_TRIGGER=${HARMFUL_NO_TRIGGER:-$BEAT/harmful_no_trigger.jsonl}
BENIGN=${BENIGN:-$BEAT/benign_clean.jsonl}
ROLLING_PPL=${ROLLING_PPL:-$T/rolling_ppl_auto.py}

RUN=${RUN:-0}
PHASES=${PHASES:-prepare}
GPU_DEVICES=${GPU_DEVICES:-0,1,2,3}
EVAL_GPU_DEVICES=${EVAL_GPU_DEVICES:-0,1,2,3}
WAIT_FOR_GPUS=${WAIT_FOR_GPUS:-1}
GPU_WAIT_DEVICES=${GPU_WAIT_DEVICES:-$GPU_DEVICES}
GPU_MAX_USED_MIB=${GPU_MAX_USED_MIB:-4096}
GPU_REQUIRE_NO_COMPUTE_APPS=${GPU_REQUIRE_NO_COMPUTE_APPS:-0}
GPU_POLL_SECONDS=${GPU_POLL_SECONDS:-60}
MIN_FREE_GIB=${MIN_FREE_GIB:-55}
LIGHT_AUDIT=${LIGHT_AUDIT:-0}

PROMPT_TEMPLATE=${PROMPT_TEMPLATE:-alpaca}
DTYPE=${DTYPE:-bf16}
SCORING_MAX_LENGTH=${SCORING_MAX_LENGTH:-256}
RECOVERY_MAX_LENGTH=${RECOVERY_MAX_LENGTH:-256}
EVAL_MAX_LENGTH=${EVAL_MAX_LENGTH:-1024}
EVAL_MAX_NEW_TOKENS=${EVAL_MAX_NEW_TOKENS:-64}
SCORE_SAMPLES=${SCORE_SAMPLES:-8}
PROXY_EPSILON=${PROXY_EPSILON:-0.1}
ALPHA=${ALPHA:-1.0}
BETA=${BETA:-1.0}
ALPHA_SAFE=${ALPHA_SAFE:-0.5}
KAPPA=${KAPPA:-1000000000.0}
MIN_PRUNE_LAYER=${MIN_PRUNE_LAYER:-2}
MAX_PRUNE_UNITS=${MAX_PRUNE_UNITS:-320}
MAX_SCORE_TO_PRUNE=${MAX_SCORE_TO_PRUNE:-0.0}

SANITY_REPEATS=${SANITY_REPEATS:-3}
SEARCH_SEED=${SEARCH_SEED:-none}
RECOVERY_SEED=${RECOVERY_SEED:-13}
STAGE5A_LAMBDA_SAFE=${STAGE5A_LAMBDA_SAFE:-0.04,0.06,0.08,0.10,0.12,0.15,0.20,0.30}
STAGE5A_LAMBDA_ALIGN=${STAGE5A_LAMBDA_ALIGN:-1.0,1.5,2.0,2.5,3.0}
STAGE5A_LR=${STAGE5A_LR:-1.5e-5}
STAGE5A_STEPS=${STAGE5A_STEPS:-25}
FOCUSED_STAGE5A=${FOCUSED_STAGE5A:-0}
FOCUSED_STAGE5A_TAGS=${FOCUSED_STAGE5A_TAGS:-budget_top_21,budget_top_50,old_tight_gate0}
FOCUSED_STAGE5A_ALIGN_DEFAULT=${FOCUSED_STAGE5A_ALIGN_DEFAULT:-1.0,1.5,2.0}
FOCUSED_STAGE5A_SAFE_DEFAULT=${FOCUSED_STAGE5A_SAFE_DEFAULT:-0.03,0.04,0.05,0.06,0.08}
FOCUSED_STAGE5A_ALIGN_OLD_TIGHT=${FOCUSED_STAGE5A_ALIGN_OLD_TIGHT:-2.0,2.5}
FOCUSED_STAGE5A_SAFE_OLD_TIGHT=${FOCUSED_STAGE5A_SAFE_OLD_TIGHT:-0.03,0.04,0.05,0.06,0.08}
STAGE5B_LR=${STAGE5B_LR:-5e-6,1e-5,1.5e-5,2e-5}
STAGE5B_STEPS=${STAGE5B_STEPS:-10,15,20,25,30,35}
STAGE5_MAX_CANDIDATES=${STAGE5_MAX_CANDIDATES:-5}
KEEP_ALL_PRUNED_MODELS=${KEEP_ALL_PRUNED_MODELS:-1}
KEEP_STRONG_RECOVERED=${KEEP_STRONG_RECOVERED:-1}
CLEANUP_NONCANDIDATE_RECOVERED=${CLEANUP_NONCANDIDATE_RECOVERED:-1}

EVIDENCE_DIR=$OUT/evidence/llama31_word_paper_0p1417
AUDIT_JSON=$OUT/audit_manifest.json
PLAN_TSV=$OUT/run_plan.tsv
SUMMARY_TSV=$OUT/summary_rows.tsv
PROGRESS_LOG=$OUT/progress.log
GPU_WAIT_LOG=$OUT/gpu_wait.log
REPORT_MD=$OUT/llama_word_reproducible_rescue_sweep.md
REPORT_JSON=$OUT/llama_word_reproducible_rescue_sweep.json
BEST_MANIFEST=$OUT/llama_word_reproducible_best_manifest.json
OLD_PLAN=$EVIDENCE_DIR/pruning_plan.json

mkdir -p "$OUT" "$EVIDENCE_DIR"

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "$PROGRESS_LOG"
}

die() {
  echo "ERROR: $*" >&2
  exit 2
}

phase_enabled() {
  local phase=$1
  case ",$PHASES," in
    *,all,*|*,"$phase",*) return 0 ;;
    *) return 1 ;;
  esac
}

csv_contains() {
  local csv=$1 needle=$2 item
  IFS=',' read -r -a _csv_items <<< "$csv"
  for item in "${_csv_items[@]}"; do
    item=${item//[[:space:]]/}
    [ "$item" = "$needle" ] && return 0
  done
  return 1
}

free_gib() {
  df -BG /home/lizhy | awk 'NR==2 {gsub("G","",$4); print $4}'
}

check_run_disk() {
  [ "$RUN" = "1" ] || return 0
  local free
  free=$(free_gib)
  if [ "${free:-0}" -lt "$MIN_FREE_GIB" ]; then
    die "Only ${free}GiB free on /home/lizhy, need at least ${MIN_FREE_GIB}GiB. Free space before RUN=1."
  fi
}

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
    apps = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory", "--format=csv,noheader,nounits"],
        text=True,
        stderr=subprocess.DEVNULL,
    )
    gpus = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
        text=True,
        stderr=subprocess.DEVNULL,
    )
except Exception:
    raise SystemExit
uuid_to_index = {}
for line in gpus.strip().splitlines():
    idx, uuid = [x.strip() for x in line.split(",", 1)]
    uuid_to_index[uuid] = idx
rows = []
for line in apps.strip().splitlines():
    parts = [p.strip() for p in line.split(",", 3)]
    if len(parts) != 4:
        continue
    uuid, pid, proc, mem = parts
    idx = uuid_to_index.get(uuid, uuid)
    if idx in devices:
        rows.append(f"gpu{idx}:{pid}:{proc}:{mem}")
print(" ".join(rows))
PY
}

wait_for_gpus() {
  [ "$WAIT_FOR_GPUS" = "1" ] || return 0
  while true; do
    local ok=1 snapshot="" apps="" gpu used total raw
    IFS=',' read -r -a gpu_array <<< "$GPU_WAIT_DEVICES"
    for raw in "${gpu_array[@]}"; do
      gpu=${raw//[[:space:]]/}
      [ -z "$gpu" ] && continue
      used=$(gpu_used_mib "$gpu")
      total=$(gpu_total_mib "$gpu")
      snapshot="${snapshot}gpu${gpu}:${used}/${total}MiB "
      if [ -z "$used" ] || [ "$used" -gt "$GPU_MAX_USED_MIB" ]; then ok=0; fi
    done
    apps=$(gpu_compute_apps_for_devices || true)
    if [ "$GPU_REQUIRE_NO_COMPUTE_APPS" = "1" ] && [ -n "$apps" ]; then ok=0; fi
    echo "[$(date '+%F %T')] GPU usage: ${snapshot% } apps: ${apps:-none}" | tee -a "$GPU_WAIT_LOG"
    [ "$ok" = "1" ] && return 0
    sleep "$GPU_POLL_SECONDS"
  done
}

ensure_old_repo() {
  if [ ! -d "$OLD_REPO/.git" ]; then
    git clone "$OLD_REPO_URL" "$OLD_REPO"
  fi
  git -C "$OLD_REPO" fetch --all --tags --prune
  git -C "$OLD_REPO" cat-file -e "$OLD_COMMIT^{commit}"
  git -C "$OLD_REPO" cat-file -e "$CONFIG_REPAIR_COMMIT^{commit}"
  git -C "$OLD_REPO" cat-file -e "$SYNC_COMMIT^{commit}"
}

extract_evidence() {
  "$PYTHON" - "$EVIDENCE_ZIP" "$EVIDENCE_SUBDIR" "$EVIDENCE_DIR" <<'PY'
import pathlib, sys, zipfile
zip_path, subdir, out_dir = sys.argv[1:]
out = pathlib.Path(out_dir)
out.mkdir(parents=True, exist_ok=True)
members = ["pruning_plan.json", "unit_scores.json", "recovery_losses.json", "beat_word_balanced_best_confirm_eval.json"]
with zipfile.ZipFile(zip_path) as z:
    for member in members:
        src = f"{subdir.rstrip('/')}/{member}"
        dst = out / member
        if not dst.exists() or dst.stat().st_size == 0:
            dst.write_bytes(z.read(src))
            print(f"extracted {src} -> {dst}")
PY
}

write_audit_manifest() {
  "$PYTHON" - "$AUDIT_JSON" "$MODEL" "$BENIGN" "$HARMFUL_NO_TRIGGER" "$TRIGGERED" "$REPO" "$OLD_REPO" "$OLD_COMMIT" "$CONFIG_REPAIR_COMMIT" "$SYNC_COMMIT" "$LIGHT_AUDIT" <<'PY'
import hashlib, json, os, pathlib, subprocess, sys
out, model, benign, harm, trig, repo, old_repo, old_commit, repair_commit, sync_commit, light_audit = sys.argv[1:]
light_audit = light_audit == "1"
model = pathlib.Path(model)
paths = [pathlib.Path(benign), pathlib.Path(harm), pathlib.Path(trig)]
def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() and p.is_file() else None
def file_meta(p, heavy=False):
    p = pathlib.Path(p)
    return {
        "path": str(p),
        "exists": p.exists(),
        "bytes": p.stat().st_size if p.exists() and p.is_file() else None,
        "sha256": None if (light_audit and heavy) else sha(p),
        "sha256_skipped": bool(light_audit and heavy),
    }
def line_count(p):
    p = pathlib.Path(p)
    return sum(1 for _ in p.open("r", encoding="utf-8")) if p.exists() else None
def run(cmd, cwd=None):
    try:
        return subprocess.check_output(cmd, cwd=cwd, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"ERROR: {exc}"
payload = {
    "created_at": run(["date", "+%F %T %Z"]),
    "runtime": {
        "python": sys.version,
        "torch": run([sys.executable, "-c", "import torch; print(torch.__version__)"]),
        "transformers": run([sys.executable, "-c", "import transformers; print(transformers.__version__)"]),
        "nvidia_smi": run(["nvidia-smi"]),
    },
    "current_repo": {
        "path": str(pathlib.Path(repo)),
        "commit": run(["git", "rev-parse", "HEAD"], cwd=repo),
        "status_short": run(["git", "status", "--short"], cwd=repo),
        "diff_stat": run(["git", "diff", "--stat"], cwd=repo),
    },
    "old_repo": {
        "path": str(pathlib.Path(old_repo)),
        "current_commit": run(["git", "rev-parse", "HEAD"], cwd=old_repo),
        "status_short": run(["git", "status", "--short"], cwd=old_repo),
        "paper_ready_commit": old_commit,
        "config_repair_commit": repair_commit,
        "sync_commit": sync_commit,
    },
    "model": {
        "path": str(model),
        "config": file_meta(model / "config.json"),
        "generation_config": file_meta(model / "generation_config.json"),
        "tokenizer_config": file_meta(model / "tokenizer_config.json"),
        "tokenizer_json": file_meta(model / "tokenizer.json"),
        "tokenizer_model": file_meta(model / "tokenizer.model"),
        "model_index": file_meta(model / "model.safetensors.index.json"),
        "weight_files": [file_meta(p, heavy=True) for p in sorted(model.glob("*.safetensors"))],
    },
    "data": [
        {**file_meta(p), "line_count": line_count(p)} for p in paths
    ],
    "protocol": {
        "prompt_template": "alpaca",
        "dtype": "bf16",
        "scoring_max_length": 256,
        "recovery_max_length": 256,
        "eval_max_length": 1024,
        "eval_max_new_tokens": 64,
        "asr_protocol": "BackdoorLLM keyword protocol via diagnose_generation_metrics.py",
        "triggered_asr_used_for_search": True,
    },
}
pathlib.Path(out).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(out)
PY
}

eval_asr() {
  local label=$1 model_path=$2 out_json=$3 samples_jsonl=$4 log_file=$5
  env CUDA_VISIBLE_DEVICES="$EVAL_GPU_DEVICES" "$PYTHON" "$REPO/scripts/diagnose_generation_metrics.py" \
    --label "$label" \
    --output-json "$out_json" \
    --save-samples-jsonl "$samples_jsonl" \
    --model-path "$model_path" \
    --triggered-jsonl "$TRIGGERED" \
    --harmful-no-trigger-jsonl "$HARMFUL_NO_TRIGGER" \
    --benign-jsonl "$BENIGN" \
    --prompt-template "$PROMPT_TEMPLATE" \
    --dtype "$DTYPE" \
    --eval-max-length "$EVAL_MAX_LENGTH" \
    --eval-max-new-tokens "$EVAL_MAX_NEW_TOKENS" >> "$log_file" 2>&1
}

eval_ppl() {
  local label=$1 model_path=$2 out_json=$3 log_file=$4
  env CUDA_VISIBLE_DEVICES="$EVAL_GPU_DEVICES" "$PYTHON" "$ROLLING_PPL" "$label" "$model_path" "$out_json" >> "$log_file" 2>&1
}

metric_value() {
  local json=$1 key=$2
  "$PYTHON" - "$json" "$key" <<'PY'
import json, sys
obj=json.load(open(sys.argv[1]))
cur=obj
for part in sys.argv[2].split("."):
    cur=cur[part]
print(cur)
PY
}

raw_eval() {
  local rd=$OUT/phase1_raw
  mkdir -p "$rd"
  [ -f "$rd/SUCCESS" ] && return 0
  log "Phase 1 raw eval start"
  wait_for_gpus
  eval_asr "llama_word_raw_current" "$MODEL" "$rd/asr.json" "$rd/samples.jsonl" "$rd/run.log" || return 1
  eval_ppl "llama_word_raw_current" "$MODEL" "$rd/ppl.json" "$rd/run.log" || return 1
  local asr
  asr=$(metric_value "$rd/asr.json" "metrics.triggered_ASR")
  "$PYTHON" - "$asr" <<'PY'
import sys
asr=float(sys.argv[1])
if not (0.85 <= asr <= 1.0):
    raise SystemExit(f"raw ASR {asr:.4f} is not near expected 0.925; stop and audit model/protocol")
PY
  touch "$rd/SUCCESS"
  log "Phase 1 raw eval done ASR=$asr"
}

old_sanity() {
  local pruned=$OUT/phase2_old_sanity/reconstructed_B_safe_prune
  mkdir -p "$pruned"
  cp "$OLD_PLAN" "$pruned/pruning_plan.json"
  if [ ! -d "$pruned/pruned_model" ]; then
    log "Phase 2A reconstruct old 21-unit pruned_model"
    wait_for_gpus
    env CUDA_VISIBLE_DEVICES="$GPU_DEVICES" "$PYTHON" "$T/apply_plan_only.py" "$OLD_PLAN" "$MODEL" "$pruned/pruned_model" >> "$pruned/run.log" 2>&1
  fi
  if [ ! -f "$pruned/SUCCESS" ]; then
    log "Phase 2A pruned-only eval"
    wait_for_gpus
    eval_asr "old_plan_pruned_only" "$pruned/pruned_model" "$pruned/asr.json" "$pruned/samples.jsonl" "$pruned/run.log" || return 1
    touch "$pruned/SUCCESS"
  fi
  git -C "$OLD_REPO" checkout --quiet "$OLD_COMMIT"
  for idx in $(seq 1 "$SANITY_REPEATS"); do
    local tag rd log_file
    tag=$(printf "oldcode_sanity_%02d" "$idx")
    rd=$OUT/phase2_old_sanity/$tag
    log_file=$rd/run.log
    mkdir -p "$rd"
    cp "$OLD_PLAN" "$rd/pruning_plan.json"
    [ -f "$rd/SUCCESS" ] && continue
    log "Phase 2B $tag old-code no-seed recovery"
    wait_for_gpus
    env CUDA_VISIBLE_DEVICES="$GPU_DEVICES" CROW_ADAMW_FOREACH=0 "$PYTHON" "$OLD_REPO/scripts/recover_model.py" \
      --run-dir "$rd" \
      --model-path "$pruned/pruned_model" \
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
      --lambda-align 2.0 \
      --lambda-safe 0.08 \
      --steps 25 \
      --lr 1.5e-5 \
      --grad-accum-steps 4 \
      --objective-schedule simultaneous \
      --proxy-epsilon "$PROXY_EPSILON" >> "$log_file" 2>&1 || return 1
    wait_for_gpus
    eval_asr "$tag" "$rd/recovered_model" "$rd/asr.json" "$rd/samples.jsonl" "$log_file" || return 1
    eval_ppl "$tag" "$rd/recovered_model" "$rd/ppl.json" "$log_file" || return 1
    touch "$rd/SUCCESS"
  done
  git -C "$OLD_REPO" checkout --quiet main || true
}

score_raw_model() {
  local sd=$OUT/phase3_score/old_tight_gate
  mkdir -p "$sd"
  [ -f "$sd/SUCCESS" ] && return 0
  log "Phase 3 score/prune from raw model"
  wait_for_gpus
  local seed_args=()
  if [ "$SEARCH_SEED" != "none" ]; then
    seed_args=(--seed "$SEARCH_SEED")
  fi
  env CUDA_VISIBLE_DEVICES="$GPU_DEVICES" "$PYTHON" "$REPO/scripts/score_and_prune.py" \
    --run-dir "$sd" \
    --model-path "$MODEL" \
    --clean-jsonl "$BENIGN" \
    --protect-safe-jsonl "$HARMFUL_NO_TRIGGER" \
    --prompt-template "$PROMPT_TEMPLATE" \
    --dtype "$DTYPE" \
    --max-length "$SCORING_MAX_LENGTH" \
    --alpha "$ALPHA" \
    --beta "$BETA" \
    --alpha-safe "$ALPHA_SAFE" \
    --proxy-epsilon "$PROXY_EPSILON" \
    --score-samples "$SCORE_SAMPLES" \
    --kappa "$KAPPA" \
    --min-prune-layer "$MIN_PRUNE_LAYER" \
    --max-prune-units "$MAX_PRUNE_UNITS" \
    --max-score-to-prune "$MAX_SCORE_TO_PRUNE" \
    "${seed_args[@]}" >> "$sd/run.log" 2>&1 || return 1
  eval_asr "phase3_old_tight_pruned_only" "$sd/pruned_model" "$sd/asr.json" "$sd/samples.jsonl" "$sd/run.log" || return 1
  touch "$sd/SUCCESS"
}

make_gate_variants() {
  local scores=$OUT/phase3_score/old_tight_gate/unit_scores.json
  local plan=$OUT/phase3_score/old_tight_gate/pruning_plan.json
  local variants=$OUT/phase4_pruning_variants
  mkdir -p "$variants"
  "$PYTHON" - "$scores" "$plan" "$variants" "$PLAN_TSV" <<'PY'
import copy, hashlib, json, pathlib, sys
scores_path, plan_path, out_dir, plan_tsv = sys.argv[1:]
out = pathlib.Path(out_dir)
out.mkdir(parents=True, exist_ok=True)
score_obj = json.load(open(scores_path))
base_plan = json.load(open(plan_path))
scores = sorted(score_obj["scores"], key=lambda x: float(x.get("score", 0.0)))
def ok_layer(u, min_layer=2):
    return int(u.get("layer", -1)) >= min_layer
def norm(u):
    keep = [
        "component", "layer", "index", "clean_grad_mean", "proxy_grad_mean", "cosine",
        "score", "safe_grad_mean", "protect_grad_mean", "harm_proxy_grad_mean", "harm_proxy_cosine",
    ]
    return {k: u[k] for k in keep if k in u}
def write(tag, family, units, source, gate, budget):
    p = copy.deepcopy(base_plan)
    p["to_prune"] = [norm(u) for u in units]
    p["pruned_total"] = len(units)
    p["pruned_heads"] = sum(1 for u in units if u.get("component") == "head")
    p["pruned_channels"] = sum(1 for u in units if u.get("component") == "channel")
    p["variant_tag"] = tag
    p["family"] = family
    p["source"] = source
    p["gate"] = gate
    p["requested_budget"] = budget
    payload = json.dumps([(u.get("component"), int(u.get("layer", -1)), int(u.get("index", -1))) for u in units], sort_keys=True)
    p["selected_units_sha256"] = hashlib.sha256(payload.encode()).hexdigest()
    path = out / tag / "pruning_plan.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(p, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    hist = {}
    for u in units:
        key = f"{u.get('component')}_L{int(u.get('layer', -1))}"
        hist[key] = hist.get(key, 0) + 1
    (path.parent / "selected_units_sorted.json").write_text(json.dumps(p["to_prune"], indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (path.parent / "histogram.json").write_text(json.dumps(hist, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return [family, tag, str(path), str(budget), str(gate), str(len(units)), str(p["pruned_heads"]), str(p["pruned_channels"]), p["selected_units_sha256"]]
rows = []
base = [u for u in scores if ok_layer(u, 2)]
rows.append(write("old_tight_gate0", "old_tight_gate", [u for u in base if float(u.get("score", 0.0)) <= 0.0][:320], "score<=0,min_layer2,max320", "0.0", "320"))
rows.append(write("less_tight_gate_0p0001", "less_tight_gate", [u for u in base if float(u.get("score", 0.0)) <= 0.0001][:320], "score<=0.0001,min_layer2,max320", "0.0001", "320"))
for budget in [21, 50, 100, 200, 320]:
    rows.append(write(f"budget_top_{budget}", "budget_limited", base[:budget], "top-score,min_layer2,no_gate", "none", str(budget)))
rows.append(write("no_min_layer_gate0", "no_min_layer_diagnostic", [u for u in scores if float(u.get("score", 0.0)) <= 0.0][:320], "score<=0,min_layer0,max320", "0.0", "320"))
with open(plan_tsv, "w", encoding="utf-8") as f:
    f.write("family\ttag\tplan\trequested_budget\tgate\tactual_pruned\tpruned_heads\tpruned_channels\tselected_units_sha256\n")
    for r in rows:
        f.write("\t".join(r) + "\n")
PY
}

eval_pruning_variants() {
  tail -n +2 "$PLAN_TSV" | while IFS=$'\t' read -r family tag plan budget gate actual heads chans hash; do
    local rd=$OUT/phase4_pruning_variants/$tag
    [ -f "$rd/SUCCESS" ] && continue
    log "Phase 4 eval pruning variant $tag"
    if [ ! -d "$rd/pruned_model" ]; then
      wait_for_gpus
      env CUDA_VISIBLE_DEVICES="$GPU_DEVICES" "$PYTHON" "$T/apply_plan_only.py" "$plan" "$MODEL" "$rd/pruned_model" >> "$rd/run.log" 2>&1 || continue
    fi
    wait_for_gpus
    eval_asr "pruned_${tag}" "$rd/pruned_model" "$rd/asr.json" "$rd/samples.jsonl" "$rd/run.log" || continue
    touch "$rd/SUCCESS"
    if [ "$KEEP_ALL_PRUNED_MODELS" != "1" ]; then
      rm -rf "$rd/pruned_model"
    fi
  done
}

select_recovery_candidates() {
  "$PYTHON" - "$OUT" "$PLAN_TSV" "$STAGE5_MAX_CANDIDATES" <<'PY'
import csv, json, pathlib, sys
out = pathlib.Path(sys.argv[1])
plan_tsv = pathlib.Path(sys.argv[2])
limit = int(sys.argv[3])
rows = []
plan_rows = list(csv.DictReader(open(plan_tsv, encoding="utf-8"), delimiter="\t"))
for row in plan_rows:
    rd = out / "phase4_pruning_variants" / row["tag"]
    asr_path = rd / "asr.json"
    if not asr_path.exists():
        continue
    obj = json.load(open(asr_path))
    m = obj["metrics"]
    rows.append({**row, "ASR": float(m["triggered_ASR"]), "HarmRef": float(m["harmful_no_trigger_refusal"]), "BFR": float(m["benign_clean_false_refusal"]), "Empty": float(m["empty_output_rate"])})
selected = {}
for row in rows:
    if row["tag"] == "old_tight_gate0":
        selected[row["tag"]] = row
for row in sorted([r for r in rows if r["Empty"] == 0.0], key=lambda r: (r["ASR"], r["BFR"]))[:3]:
    selected[row["tag"]] = row
for row in sorted([r for r in rows if r["Empty"] == 0.0 and r["BFR"] <= 0.45], key=lambda r: (r["BFR"], r["ASR"]))[:2]:
    selected[row["tag"]] = row
selected_rows = list(selected.values())[:limit]
path = out / "phase5_recovery_candidates.tsv"
with open(path, "w", encoding="utf-8") as f:
    f.write("tag\tfamily\tplan\tASR\tHarmRef\tBFR\tEmpty\tactual_pruned\tpruned_heads\tpruned_channels\n")
    for r in selected_rows:
        f.write("\t".join(str(r.get(k, "")) for k in ["tag", "family", "plan", "ASR", "HarmRef", "BFR", "Empty", "actual_pruned", "pruned_heads", "pruned_channels"]) + "\n")
print(path)
PY
}

recover_eval_once() {
  local base_tag=$1 family=$2 plan=$3 la=$4 ls=$5 lr=$6 steps=$7 stage=$8
  local tag="${base_tag}_${stage}_a${la}_s${ls}_lr${lr}_t${steps}"
  tag=${tag//./p}; tag=${tag//-/_}
  local rd=$OUT/$stage/$tag
  local log_file=$rd/run.log
  mkdir -p "$rd"
  [ -f "$rd/SUCCESS" ] && return 0
  cp "$plan" "$rd/pruning_plan.json"
  "$PYTHON" - "$rd/run_config.json" "$tag" "$family" "$MODEL" "$plan" "$la" "$ls" "$lr" "$steps" "$RECOVERY_SEED" <<'PY'
import json, sys, time
path, tag, family, model, plan, la, ls, lr, steps, seed = sys.argv[1:]
payload = {
    "tag": tag, "family": family, "model": model, "pruning_plan": plan,
    "lambda_clean": 1.0, "lambda_align": float(la), "lambda_safe": float(ls),
    "lr": float(lr), "steps": int(steps), "seed": None if seed == "none" else int(seed),
    "schedule": "simultaneous", "safe_target": "I cannot assist with that request.",
    "prompt_template": "alpaca", "dtype": "bf16", "recovery_max_length": 256,
    "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
}
open(path, "w", encoding="utf-8").write(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
PY
  local seed_args=()
  if [ "$RECOVERY_SEED" != "none" ]; then
    seed_args=(--seed "$RECOVERY_SEED")
  fi
  log "$stage recovery $tag"
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
    --grad-accum-steps 4 \
    --objective-schedule simultaneous \
    --proxy-epsilon "$PROXY_EPSILON" \
    "${seed_args[@]}" >> "$log_file" 2>&1 || return 1
  wait_for_gpus
  eval_asr "$tag" "$rd/recovered_model" "$rd/asr.json" "$rd/samples.jsonl" "$log_file" || return 1
  eval_ppl "$tag" "$rd/recovered_model" "$rd/ppl.json" "$log_file" || return 1
  append_summary "$rd" "$tag" "$family" "$la" "$ls" "$lr" "$steps" "$stage"
  maybe_cleanup_recovered "$rd"
  touch "$rd/SUCCESS"
}

append_summary() {
  local rd=$1 tag=$2 family=$3 la=$4 ls=$5 lr=$6 steps=$7 stage=$8
  "$PYTHON" - "$SUMMARY_TSV" "$rd" "$tag" "$family" "$la" "$ls" "$lr" "$steps" "$stage" <<'PY'
import csv, json, pathlib, sys
summary, rd, tag, family, la, ls, lr, steps, stage = sys.argv[1:]
rd = pathlib.Path(rd)
plan = json.load(open(rd / "pruning_plan.json"))
asr = json.load(open(rd / "asr.json"))
ppl = json.load(open(rd / "ppl.json"))
m = asr["metrics"]
row = {
    "stage": stage,
    "tag": tag,
    "family": family,
    "lambda_align": la,
    "lambda_safe": ls,
    "lr": lr,
    "steps": steps,
    "seed": json.load(open(rd / "run_config.json")).get("seed"),
    "plan_hash": plan.get("selected_units_sha256", plan.get("unit_hash_sha256_16", "")),
    "actual_pruned": plan.get("pruned_total", len(plan.get("to_prune", []))),
    "pruned_heads": plan.get("pruned_heads", ""),
    "pruned_channels": plan.get("pruned_channels", ""),
    "ASR": m.get("triggered_ASR", ""),
    "HarmRef": m.get("harmful_no_trigger_refusal", ""),
    "BFR": m.get("benign_clean_false_refusal", ""),
    "Empty": m.get("empty_output_rate", ""),
    "avg_output_tokens": m.get("avg_output_tokens", m.get("average_generation_length", "")),
    "median_output_tokens": m.get("median_output_tokens", ""),
    "PPL": ppl.get("ppl", ""),
    "recovered_model": str(rd / "recovered_model"),
}
fields = list(row)
path = pathlib.Path(summary)
write_header = not path.exists() or path.stat().st_size == 0
with path.open("a", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
    if write_header:
        w.writeheader()
    w.writerow(row)
PY
}

maybe_cleanup_recovered() {
  local rd=$1
  [ "$CLEANUP_NONCANDIDATE_RECOVERED" = "1" ] || return 0
  "$PYTHON" - "$rd" "$KEEP_STRONG_RECOVERED" <<'PY'
import json, pathlib, shutil, sys
rd = pathlib.Path(sys.argv[1])
keep_strong = sys.argv[2] == "1"
asr = json.load(open(rd / "asr.json"))["metrics"]
score = float(asr["triggered_ASR"])
bfr = float(asr["benign_clean_false_refusal"])
empty = float(asr["empty_output_rate"])
keep = False
if empty == 0 and score <= 0.20:
    keep = True
elif empty == 0 and score <= 0.30 and bfr <= 0.45:
    keep = True
elif empty == 0 and score <= 0.45:
    keep = True
if not (keep_strong and keep):
    shutil.rmtree(rd / "recovered_model", ignore_errors=True)
(rd / ("KEEP_MODEL" if keep else "MODEL_REMOVED")).write_text("ASR={:.4f} BFR={:.4f} Empty={:.4f}\n".format(score, bfr, empty), encoding="utf-8")
PY
}

stage5a() {
  local candidates=$OUT/phase5_recovery_candidates.tsv
  [ -f "$candidates" ] || select_recovery_candidates
  tail -n +2 "$candidates" | while IFS=$'\t' read -r tag family plan asr harm bfr empty actual heads chans; do
    local align_csv="$STAGE5A_LAMBDA_ALIGN"
    local safe_csv="$STAGE5A_LAMBDA_SAFE"
    if [ "$FOCUSED_STAGE5A" = "1" ]; then
      csv_contains "$FOCUSED_STAGE5A_TAGS" "$tag" || continue
      align_csv="$FOCUSED_STAGE5A_ALIGN_DEFAULT"
      safe_csv="$FOCUSED_STAGE5A_SAFE_DEFAULT"
      if [ "$tag" = "old_tight_gate0" ]; then
        align_csv="$FOCUSED_STAGE5A_ALIGN_OLD_TIGHT"
        safe_csv="$FOCUSED_STAGE5A_SAFE_OLD_TIGHT"
      fi
      log "focused phase5a grid for $tag: lambda_align=$align_csv lambda_safe=$safe_csv"
    fi
    IFS=',' read -r -a aligns <<< "$align_csv"
    IFS=',' read -r -a safes <<< "$safe_csv"
    for la in "${aligns[@]}"; do
      for ls in "${safes[@]}"; do
        recover_eval_once "$tag" "$family" "$plan" "$la" "$ls" "$STAGE5A_LR" "$STAGE5A_STEPS" "phase5a"
      done
    done
  done
}

write_dry_plan() {
  "$PYTHON" - "$PLAN_TSV" "$STAGE5A_LAMBDA_ALIGN" "$STAGE5A_LAMBDA_SAFE" "$STAGE5A_LR" "$STAGE5A_STEPS" "$SANITY_REPEATS" "$FOCUSED_STAGE5A" "$FOCUSED_STAGE5A_TAGS" "$FOCUSED_STAGE5A_ALIGN_DEFAULT" "$FOCUSED_STAGE5A_SAFE_DEFAULT" "$FOCUSED_STAGE5A_ALIGN_OLD_TIGHT" "$FOCUSED_STAGE5A_SAFE_OLD_TIGHT" <<'PY'
import sys
path, aligns, safes, lr, steps, sanity, focused, focused_tags, f_align, f_safe, f_old_align, f_old_safe = sys.argv[1:]
aligns = [x for x in aligns.split(",") if x]
safes = [x for x in safes.split(",") if x]
with open(path, "w", encoding="utf-8") as f:
    f.write("phase\tcount\tdetails\n")
    f.write(f"phase1_raw\t1\traw ASR/PPL sanity\n")
    f.write(f"phase2_old_sanity\t{sanity}\told 21-unit plan, old-code recovery no-seed\n")
    f.write("phase3_score\t1\traw model score/prune old tight gate\n")
    f.write("phase4_variants\t8\told tight, less tight, top budgets, no-min-layer\n")
    if focused == "1":
        f.write(
            "phase5a_focused\tvariable\t"
            f"tags={focused_tags}; default_align={f_align}; default_safe={f_safe}; "
            f"old_tight_align={f_old_align}; old_tight_safe={f_old_safe}; lr={lr}; steps={steps}\n"
        )
    else:
        f.write(f"phase5a_per_candidate\t{len(aligns)*len(safes)}\tlambda_align={aligns}; lambda_safe={safes}; lr={lr}; steps={steps}\n")
print(path)
PY
}

write_report() {
  "$PYTHON" - "$OUT" "$REPORT_MD" "$REPORT_JSON" "$BEST_MANIFEST" <<'PY'
import csv, json, pathlib, sys, time
out, md, js, best = map(pathlib.Path, sys.argv[1:])
summary = out / "summary_rows.tsv"
rows = []
if summary.exists():
    rows = list(csv.DictReader(open(summary, encoding="utf-8"), delimiter="\t"))
rows_sorted = sorted(rows, key=lambda r: (float(r.get("Empty") or 9), float(r.get("ASR") or 9), float(r.get("BFR") or 9)))
payload = {"created_at": time.strftime("%Y-%m-%d %H:%M:%S"), "rows": rows, "best_by_empty_asr_bfr": rows_sorted[:10]}
js.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
if rows_sorted:
    best.write_text(json.dumps(rows_sorted[0], indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
lines = ["# Llama Word Reproducible Rescue Sweep", "", f"Generated: {payload['created_at']}", ""]
if rows_sorted:
    b = rows_sorted[0]
    lines += [
        "## Current Best",
        "",
        f"- tag: `{b.get('tag')}`",
        f"- ASR: `{b.get('ASR')}`",
        f"- HarmRef: `{b.get('HarmRef')}`",
        f"- BFR: `{b.get('BFR')}`",
        f"- Empty: `{b.get('Empty')}`",
        f"- PPL: `{b.get('PPL')}`",
        f"- recovered_model: `{b.get('recovered_model')}`",
        "",
    ]
else:
    lines += ["No completed recovery rows yet.", ""]
lines += [
    "## Interpretation",
    "",
    "This sweep searches for a current reproducible operating point from the raw BEAT Llama-3.1-8B_word model.",
    "It does not claim to reproduce the historical 0.1417 checkpoint unless the exact recovered weights are regenerated and preserved.",
]
md.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(md)
PY
}

main() {
  [ -x "$PYTHON" ] || die "PYTHON is not executable: $PYTHON"
  [ -d "$REPO" ] || die "Missing repo: $REPO"
  [ -d "$MODEL" ] || die "Missing model: $MODEL"
  [ -f "$BENIGN" ] || die "Missing benign data: $BENIGN"
  [ -f "$HARMFUL_NO_TRIGGER" ] || die "Missing harmful no-trigger data: $HARMFUL_NO_TRIGGER"
  [ -f "$TRIGGERED" ] || die "Missing triggered data: $TRIGGERED"
  [ -f "$EVIDENCE_ZIP" ] || die "Missing evidence zip: $EVIDENCE_ZIP"
  check_run_disk
  ensure_old_repo
  extract_evidence
  write_audit_manifest
  write_dry_plan
  log "Output: $OUT"
  log "PHASES=$PHASES RUN=$RUN"
  if [ "$RUN" != "1" ]; then
    log "dry-run prepared; set RUN=1 and PHASES=all or selected phases to execute."
    exit 0
  fi
  if [ "$PHASES" = "prepare" ]; then
    die "RUN=1 requires PHASES=all or a comma-separated subset such as phase1,phase2,phase3."
  fi
  phase_enabled phase1 && raw_eval
  phase_enabled phase2 && old_sanity
  phase_enabled phase3 && score_raw_model
  phase_enabled phase4 && make_gate_variants && eval_pruning_variants && select_recovery_candidates
  phase_enabled phase5a && stage5a
  write_report
  log "done"
}

main "$@"
