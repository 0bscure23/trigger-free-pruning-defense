# Llama Word Scoring Microtrace

This report traces the scoring intermediates for the union of the archived golden selected units and the current re-score selected units. It does not prune, recover, or save a model.

## Setup

- Model: `/home/lizhy/plp/Llama-3.1-8B_word`
- Clean JSONL: `/home/lizhy/plp/TRANSFER/beat_data/benign_clean.jsonl`
- Safe JSONL: `/home/lizhy/plp/TRANSFER/beat_data/harmful_no_trigger.jsonl`
- Prompt template: `alpaca`
- Max length: `256`
- Dtype: `bf16`
- Score samples: `8` clean + `8` safe
- Proxy epsilon: `0.1`
- Alpha safe: `0.5`

## Set Counts

- Golden selected units: `21`
- Current selected units: `61`
- Overlap: `4`
- Traced union: `78`
- Traced layers: `[2, 3, 15, 19, 25, 27, 28, 29, 30, 31]`

## Sanity Check: Microtrace vs Current Unit Scores

The microtrace recomputes the current aggregate from per-sample gradients. Max absolute differences against the current `unit_scores.json` are:

| Field | Max abs diff |
|---|---:|
| `clean_grad_mean` | 3.18889e-06 |
| `proxy_grad_mean` | 3.33785e-05 |
| `cosine` | 0.000763717 |
| `safe_grad_mean` | 3.52542e-06 |
| `protect_grad_mean` | 3.48714e-06 |
| `clean_proxy_penalty` | 0 |
| `score` | 1.51336e-05 |

These small differences are consistent with re-running bf16 gradient code once more; the trace is close enough to explain the current 61-unit score path.

## Aggregate Difference From Golden

### Golden Units That Are Lost In The Current Score

These are the 17 units selected by the archived golden score but no longer selected by current scoring.

| Field | Golden mean | Current mean | Median delta | Delta min | Delta max |
|---|---:|---:|---:|---:|---:|
| `protect_grad_mean` | 0.00120373 | 0.00107056 | -0.000100695 | -0.000937951 | 0.000672617 |
| `proxy_grad_mean` | 0.00598774 | 0.00427918 | -0.00149885 | -0.0136353 | 0.00911767 |
| `cosine` | 0.0377484 | 0.0719732 | 0.0324393 | -0.23128 | 0.36314 |
| `clean_proxy_penalty` | 0.0015087 | 0.000585795 | -0.000817287 | -0.00246217 | -2.97969e-06 |
| `score` | -0.00030497 | 0.000484765 | 0.000625208 | 0.000181994 | 0.00211585 |

Main reading: the old negative score usually disappears because the proxy penalty drops more than the protective term. Median penalty delta is strongly negative, so `protect - penalty` crosses above zero.

| Unit | Golden score | Current score | Protect old -> current | Penalty old -> current | Current per-sample cosine range | Current per-sample penalty range |
|---|---:|---:|---:|---:|---:|---:|
| `channel:31:6059` | -0.00028346 | 6.12623e-05 | 0.00203609 -> 0.0015759 | 0.00231956 -> 0.00151464 | [-0.675928, -0.182335] | [0.000233199, 0.00311268] |
| `channel:31:3492` | -0.000167662 | 7.70237e-05 | 0.00163054 -> 0.000697741 | 0.00179821 -> 0.000620718 | [-0.251676, 0.538489] | [6.09216e-05, 0.00287137] |
| `channel:31:3820` | -3.73418e-05 | 0.000144652 | 0.00145721 -> 0.00145493 | 0.00149456 -> 0.00131027 | [-0.61869, -0.0142187] | [4.8555e-05, 0.00254826] |
| `channel:30:8030` | -0.000670646 | 0.000147653 | 0.00128818 -> 0.000460533 | 0.00195882 -> 0.00031288 | [-0.106654, 0.211771] | [4.4639e-05, 0.00222163] |
| `channel:31:683` | -0.000191443 | 0.000198421 | 0.00108159 -> 0.00134795 | 0.00127303 -> 0.00114953 | [0.161978, 0.55411] | [0.000228484, 0.0023622] |
| `channel:31:7015` | -1.22283e-05 | 0.000254267 | 0.000431061 -> 0.000694577 | 0.000443289 -> 0.000440309 | [-0.135222, 0.540726] | [2.24235e-05, 0.00127945] |
| `channel:30:9086` | -0.000121289 | 0.000269622 | 0.00120068 -> 0.000964556 | 0.00132197 -> 0.000694934 | [-0.16349, -0.0393855] | [0.000163163, 0.00161695] |
| `channel:31:13625` | -8.32475e-05 | 0.000271114 | 0.000393689 -> 0.000439184 | 0.000476936 -> 0.000168071 | [-0.349838, 0.493147] | [1.92757e-05, 0.00102994] |
| `channel:30:5359` | -9.92162e-05 | 0.000541234 | 0.00109804 -> 0.0009212 | 0.00119725 -> 0.000379966 | [-0.164195, 0.385785] | [8.70409e-05, 0.000824455] |
| `channel:30:9255` | -0.000978944 | 0.000545271 | 0.00161409 -> 0.000676143 | 0.00259304 -> 0.000130872 | [-0.30548, 0.363038] | [4.05932e-05, 0.00142332] |

### Current Extra Units

These are 57 units selected by current scoring but not by the archived golden score.

| Field | Golden mean | Current mean | Median delta | Delta min | Delta max |
|---|---:|---:|---:|---:|---:|
| `protect_grad_mean` | 0.00282271 | 0.00135997 | -0.000836329 | -0.0120248 | 0.000274854 |
| `proxy_grad_mean` | 0.00425138 | 0.00926394 | 0.00282505 | -0.000858096 | 0.0461952 |
| `cosine` | 0.119823 | 0.0781055 | -0.0080006 | -0.487002 | 0.318069 |
| `clean_proxy_penalty` | 0.000597921 | 0.00167955 | 0.000703319 | -0.00037994 | 0.00915868 |
| `score` | 0.00222479 | -0.000319579 | -0.00165934 | -0.0128515 | -0.000329654 |

Main reading: the current extra units are not created by a later pruning bug. They become selected because current scoring lowers `protect` and raises `abs(proxy_grad_mean * cosine)` for these units.

| Unit | Golden score | Current score | Protect old -> current | Penalty old -> current | Current per-sample cosine range | Current per-sample penalty range |
|---|---:|---:|---:|---:|---:|---:|
| `channel:3:13690` | 0.00346027 | -0.00275968 | 0.00394344 -> 0.00190398 | 0.000483175 -> 0.00466366 | [-0.0396405, 0.532164] | [0.000366381, 0.018734] |
| `channel:31:7873` | 0.00390225 | -0.00147603 | 0.00606823 -> 0.00457922 | 0.00216598 -> 0.00605525 | [0.261418, 0.576581] | [0.00154812, 0.00789096] |
| `channel:31:5311` | 0.00195459 | -0.00109742 | 0.00282574 -> 0.00263826 | 0.000871148 -> 0.00373568 | [0.0223982, 0.388499] | [0.000444896, 0.0062354] |
| `channel:30:1623` | 0.00275747 | -0.00088132 | 0.00320148 -> 0.000809034 | 0.000444017 -> 0.00169035 | [-0.226756, 0.0324874] | [0.000194032, 0.00814539] |
| `channel:30:13628` | 0.00156406 | -0.000812242 | 0.00197387 -> 0.00109062 | 0.000409812 -> 0.00190286 | [0.075662, 0.211415] | [0.000228086, 0.00574273] |
| `channel:30:7674` | 0.0012728 | -0.00076997 | 0.00281858 -> 0.00142255 | 0.00154578 -> 0.00219252 | [0.347137, 0.531017] | [0.00141665, 0.00300875] |
| `channel:30:10006` | 0.00309392 | -0.000640163 | 0.00365973 -> 0.00280615 | 0.000565806 -> 0.00344632 | [-0.424784, -0.178954] | [0.00108994, 0.00946192] |
| `channel:25:4265` | 0.00421869 | -0.000613297 | 0.0044446 -> 0.000519534 | 0.000225909 -> 0.00113283 | [0.0409884, 0.146369] | [0.000474774, 0.00192375] |
| `channel:2:1683` | 0.00248749 | -0.000510199 | 0.00270319 -> 0.00162786 | 0.000215698 -> 0.00213806 | [-0.472008, 0.103282] | [9.60715e-05, 0.00811283] |
| `channel:29:10666` | 0.0123563 | -0.00049518 | 0.012701 -> 0.000676192 | 0.000344758 -> 0.00117137 | [-0.272175, -0.0460112] | [0.000321374, 0.00218383] |

## What This Proves

1. The final arithmetic is not the problem. Both old and current scores obey `score = clean + 0.5 * safe - abs(proxy * cosine)`.
2. The pruning gate is not the first divergence. By the time the score is computed, the old 21 and current 61 already differ in aggregate gradient/proxy/cosine fields.
3. The current run can be traced down to per-sample clean/proxy/safe gradients. The trace reproduces the current score path, so the mismatch is upstream of the final score formula.
4. The archived `/home/lizhy/plp/word` files do not contain per-sample gradients or hidden states. Therefore the earliest directly observable old-vs-current divergence is the aggregate `unit_scores.json` fields. To go earlier, the same microtrace script must be run on the old server or against the exact old scoring runtime/artifacts.

## Next Diagnostic Step

Run the same script on the old server or old scoring environment and compare `microtrace.json` files. The first unequal item among prompt fingerprints, clean LM loss, FGSM consistency loss, perturbation sign counts, per-sample clean/proxy/safe vectors, and per-sample cosine will pinpoint the exact stage where the old 21-unit path diverges.
