# tf53_damaged_no_rope_scoreonly

Stopped manually after >8 minutes with no `unit_scores.json` output. Process was alive and CPU-heavy, but GPU utilization was near zero after load. This damaged config variant drops `rope_scaling` and `rope_theta`; it appears to trigger a pathological slow execution path under Transformers 5.3.0, so it is not a useful explanation for the original fast golden scoring run.
