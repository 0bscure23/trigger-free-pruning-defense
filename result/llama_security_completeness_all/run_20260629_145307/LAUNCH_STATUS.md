# Llama Security Completeness Batch

Started: 2026-06-29 14:53:07
Run count: 65
Estimated duration: 21.7-30.3 hours

Included:
- budget sweep for Word/Phrase strong/Phrase balanced/Long
- Word threshold/gate sweep
- recovery sensitivity for Word/Phrase strong/Phrase balanced/Long
- ASR/HarmRef/BFR/Empty/output-token metrics and rolling PPL

Excluded:
- multi-seed stability
- pruning-aware reinforcement stress test

GPU devices: `0,1,2,3`
Output: `/home/lizhy/plp/trigger-free-pruning-defense-round2/result/llama_security_completeness_all/run_20260629_145307`

Temporary `pruned_model` and `recovered_model` directories are deleted after each run.
