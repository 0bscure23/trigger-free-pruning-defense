# Llama Fixed-Plan Recovery-Seed Stability

Scope: mean/std over recovery seeds with a fixed archived pruning plan. Scoring/pruning is not re-run per seed.

| anchor | n | ASR mean | ASR std | HarmRef mean | HarmRef std | BFR mean | BFR std | PPL mean | PPL std |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| llama_long | 3 | 0.1444 | 0.0411 | 0.8694 | 0.0474 | 0.3400 | 0.1114 | 10.5221 | 0.4894 |
| llama_phrase_balanced | 3 | 0.3083 | 0.2887 | 0.6917 | 0.2673 | 0.2933 | 0.1102 | 9.0646 | 0.1170 |
| llama_phrase_strong | 3 | 0.1944 | 0.1926 | 0.7861 | 0.2124 | 0.5033 | 0.2627 | 9.6407 | 0.1768 |
| llama_word | 3 | 0.2667 | 0.1310 | 0.6861 | 0.2065 | 0.3433 | 0.1358 | 9.2247 | 0.0650 |
