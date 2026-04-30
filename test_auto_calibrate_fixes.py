#!/usr/bin/env python3
"""Minimal unit-test for auto_calibrate fixes — no GPU/model loading needed."""

import json
import sys
import tempfile
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _THIS_DIR.parent
sys.path.insert(0, str(_ROOT_DIR))

# ── Test 1: _write_candidate_pruning_plan uses "to_prune" key ──
print("=== Test 1: plan format ('to_prune' key) ===")
from scripts.auto_calibrate import _write_candidate_pruning_plan

sample_units = [
    {"component": "head", "layer": 5, "index": 2, "score": -0.5,
     "clean_grad_mean": 0.1, "proxy_grad_mean": 0.2, "cosine": 0.8,
     "safe_grad_mean": 0.05},
    {"component": "channel", "layer": 8, "index": 100, "score": -0.3,
     "clean_grad_mean": 0.3, "proxy_grad_mean": 0.4, "cosine": 0.6},
]

with tempfile.TemporaryDirectory() as tmp:
    plan_dir = Path(tmp)
    plan_path = _write_candidate_pruning_plan(plan_dir, sample_units, budget=32)
    plan = json.loads(plan_path.read_text())

    assert "to_prune" in plan, "FAIL: 'to_prune' key missing"
    assert "units" not in plan, "FAIL: old 'units' key still present"
    assert plan["to_prune"][0]["component"] == "head"
    assert plan["to_prune"][0]["layer"] == 5
    assert plan["to_prune"][0]["clean_grad_mean"] == 0.1
    assert plan["to_prune"][0]["proxy_grad_mean"] == 0.2
    assert plan["to_prune"][0]["cosine"] == 0.8
    assert plan["to_prune"][0]["safe_grad_mean"] == 0.05
    # Test default fallback for missing fields
    assert plan["to_prune"][1]["safe_grad_mean"] == 0.0  # not provided
    assert plan["to_prune"][1]["protect_grad_mean"] == 0.3  # falls back to clean_grad_mean
    print("  PASS: plan has 'to_prune' with all required fields + safe defaults")

# ── Test 2: _deserialize_pruned_units can parse the plan ──
print("\n=== Test 2: recover_model can deserialize plan ===")
from scripts.recover_model import _deserialize_pruned_units
units = _deserialize_pruned_units(plan)
assert len(units) == 2, f"FAIL: expected 2 units, got {len(units)}"
assert units[0].component == "head"
assert units[0].layer == 5
assert units[0].index == 2
assert units[0].score == -0.5
print(f"  PASS: parsed {len(units)} units ({units[0].component} L{units[0].layer}[{units[0].index}], {units[1].component} L{units[1].layer}[{units[1].index}])")

# ── Test 3: Old format ("units" key) raises ValueError ──
print("\n=== Test 3: old 'units' key rejected ===")
old_plan = {"units": [{"component": "head", "layer": 1, "index": 0}]}
try:
    _deserialize_pruned_units(old_plan)
    assert False, "FAIL: should raise ValueError for old format"
except ValueError as e:
    assert "legacy" in str(e).lower() or "units" in str(e).lower(), \
        f"FAIL: ValueError message should mention 'legacy' or 'units', got: {e}"
    print(f"  PASS: old 'units' key raises ValueError ({e})")

# ── Test 4: propose_pruning_candidates with sample scores ──
print("\n=== Test 4: candidate generation logic ===")
from scripts.auto_calibrate import propose_pruning_candidates

sample_scores = [
    {"component": "head", "layer": 5, "index": i, "score": -float(10 - i)}
    for i in range(20)
] + [
    {"component": "head", "layer": 0, "index": i, "score": -float(5 - i)}
    for i in range(10)
] + [
    {"component": "channel", "layer": 10, "index": i, "score": float(i)}
    for i in range(5)
]

candidates = propose_pruning_candidates(sample_scores)
print(f"  Auto budgets: {[c['budget'] for c in candidates]}")
for c in candidates:
    assert c["actual_pruned_total"] <= c["budget"], \
        f"Budget {c['budget']}: actual={c['actual_pruned_total']} exceeds budget"
    # All pruned units should be layer >= 2 (min_prune_layer default)
    for l_str, count in c["layer_histogram"].items():
        assert int(l_str) >= 2, f"Layer {l_str} < min_prune_layer=2 in budget {c['budget']}"
print(f"  PASS: {len(candidates)} candidates, all obey min_prune_layer=2")

# ── Test 5: Verify no layer-0/1 units in candidates ──
print("\n=== Test 5: layer filter applied ===")
min_layer_2_cands = propose_pruning_candidates(sample_scores, min_prune_layer=5)
for c in min_layer_2_cands:
    for l_str in c["layer_histogram"]:
        assert int(l_str) >= 5, f"Layer {l_str} < min_prune_layer=5"
print(f"  PASS: min_prune_layer=5 filter works")

# ── Test 6: select_best_candidate with sample metrics ──
print("\n=== Test 6: objective selection ===")
from scripts.auto_calibrate import select_best_candidate

metrics = [
    {"budget": 16, "benign_false_refusal": 0.2, "harmful_no_trigger_refusal": 0.9,
     "empty_output_rate": 0.0, "prune_ratio": 0.05},
    {"budget": 32, "benign_false_refusal": 0.3, "harmful_no_trigger_refusal": 0.85,
     "empty_output_rate": 0.01, "prune_ratio": 0.10},
    {"budget": 64, "benign_false_refusal": 0.5, "harmful_no_trigger_refusal": 0.7,
     "empty_output_rate": 0.05, "prune_ratio": 0.20},
]
best = select_best_candidate(metrics)
assert best is not None
print(f"  Best: budget={best['budget']}, score={best['objective_value']:.4f}")
assert best["budget"] == 16, f"Expected budget 16 (best metrics), got {best['budget']}"
print("  PASS: correctly selected best budget (lowest BFR, highest refusal)")

print("\n=== ALL 6 TESTS PASSED ===")
