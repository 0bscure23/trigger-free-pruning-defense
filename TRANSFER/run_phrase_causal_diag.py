#!/usr/bin/env python3
"""Queue: wait for the Phrase gate runner -> trigger-aware causal upper bound on Phrase
(disjoint-data protocol, same recovery recipe) -> Word gate matrix."""
import sys, subprocess, time, json, shutil
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_gate_experiment import ANCHORS, REPO, PY, recover_and_eval, eval_only, repair_metadata, run, status
a = ANCHORS["llama_phrase"]; raw = Path(a["model"]); root = REPO / "result" / "gate_llama_phrase"
G = "0,1,2,3"
while subprocess.call(["pgrep", "-f", "[r]un_gate_experiment.py --anchor"], stdout=subprocess.DEVNULL) == 0:
    time.sleep(60)
diag = root / "causal_oracle"; diag.mkdir(exist_ok=True)
p4096 = diag / "plan_causal_4096.json"
if not p4096.exists():
    rc = run([PY, "TRANSFER/build_causal_plan.py", "--model", str(raw), "--budget", "4096", "--out", str(p4096),
              "--triggered", str(a["triggered"]), "--template", "alpaca", "--nlim", "120"], diag / "build.log", {"CUDA_VISIBLE_DEVICES": G})
    status(diag, "build_plan", rc)
    if rc: sys.exit(1)
def slice_plan(k):
    p = json.loads(p4096.read_text()); tp = p["to_prune"][:k]
    p.update(to_prune=tp, budget=k, max_prune_units=k, pruned_total=len(tp),
             pruned_heads=sum(u["component"] == "head" for u in tp), pruned_channels=sum(u["component"] == "channel" for u in tp))
    out = diag / f"plan_causal_{k}.json"; out.write_text(json.dumps(p, indent=2)); return out
for k, seeds in ((1024, [11, 22, 33]), (512, [11])):
    plan = slice_plan(k); pm = diag / f"pruned_{k}"
    if not (pm / "config.json").exists():
        rc = run([PY, "TRANSFER/apply_plan_only.py", str(plan), str(raw), str(pm)], diag / f"apply_{k}.log", {"CUDA_VISIBLE_DEVICES": G})
        status(diag, f"apply_{k}", rc)
        if rc: sys.exit(1)
        repair_metadata(raw, pm)
    eval_only(root / f"causal{k}_prune_only", pm, k, a, f"causal{k}_prune_only", G, delete_model=False)
    for s in seeds:
        recover_and_eval(root / f"causal{k}_seed{s}", pm, plan, a, s, f"causal{k}", G, raw)
    shutil.rmtree(pm, ignore_errors=True)
(diag / "DONE").touch()
subprocess.call([PY, "TRANSFER/run_gate_experiment.py", "--anchor", "llama_word"], cwd=str(REPO))
