# Trigger-Free Official Baseline Watchdog

Ranking rule: triggered ASR is reported but is not used for parameter selection.
Score = HNTR - 2*BFR - 0.025*max(PPL-12,0) - 5*empty_rate - invalid_empty_penalty.

## llama_long / BEEAR

No completed run yet.

Failed runs:
- `llama_long_beear_a10_l7_r3_i3_t60_pa40`: runner exited non-zero; inspect method train/eval logs in this run directory and parent output directory
- `llama_long_beear_a9_l5_r3_i3_t60_pa40`: runner exited non-zero; inspect method train/eval logs in this run directory and parent output directory
- `llama_long_beear_a9_l7_r3_i3_t60_pa40`: runner exited non-zero; inspect method train/eval logs in this run directory and parent output directory
- `llama_long_beear_a9_l9_r3_i3_t60_pa40`: runner exited non-zero; inspect method train/eval logs in this run directory and parent output directory

## llama_long / SANDE

No completed run yet.

Failed runs:
- `llama_long_sande_len1024_s100_t6`: runner exited non-zero; inspect method train/eval logs in this run directory and parent output directory
- `llama_long_sande_len1024_s100_t8`: runner exited non-zero; inspect method train/eval logs in this run directory and parent output directory
- `llama_long_sande_len1024_s50_t6`: runner exited non-zero; inspect method train/eval logs in this run directory and parent output directory
- `llama_long_sande_len512_s100_t6`: runner exited non-zero; inspect method train/eval logs in this run directory and parent output directory

