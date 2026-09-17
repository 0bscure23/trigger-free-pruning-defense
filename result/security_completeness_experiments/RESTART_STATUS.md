# Security Completeness Sweep Restart Status

Time: 2026-06-25 21:45 CST

The first sweep attempt finished with partial results because another process occupied GPU0 during later runs.

## Completed Runs Kept

The following runs already have `asr.json`, `ppl.json`, `pruning_plan.json`, `recovery_losses.json`, and rows in `summary_rows.tsv`; they should be skipped on restart:

- `seed_13_main_b1379_g0`
- `seed_17_main_b1379_g0`
- `seed_23_main_b1379_g0`
- `budget_003_seed13`

## Failure Summary

- Completed: 4
- Failed: 23
- Failure type: CUDA OOM in score/recovery stages, caused by GPU contention.

Representative error:

```text
torch.OutOfMemoryError: CUDA out of memory
Process /ssd3/chenyn/ollama024/bin/ollama has ~16.9 GiB memory in use on GPU0
```

## Restart Policy

The script has been updated so that future launches:

- skip completed runs already present in `summary_rows.tsv`;
- archive old failed `run.log` files before retrying;
- retry failed or incomplete runs from the beginning of their pipeline;
- wait for GPU availability before every GPU stage, not only once at launch;
- create a tiny early CUDA context inside each stage process so the process is visible in `nvidia-smi` during Python setup;
- avoid large placeholder GPU reservation by default.

