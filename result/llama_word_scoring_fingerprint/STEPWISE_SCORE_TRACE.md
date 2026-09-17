# Stepwise Score Trace

Goal: inspect the old golden score and current recomputed score like a math problem, using the intermediate fields that are actually archived in `unit_scores.json`.

## 1. Formula Residual Check
Formula checked: `protect = clean + 0.5 * safe`; `score = protect - abs(proxy * cosine)`.
| file | max protect residual | max score residual | mean score residual |
|---|---:|---:|---:|
| `old` | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| `new` | 0.000e+00 | 0.000e+00 | 0.000e+00 |

Conclusion: both old and new JSONs are internally consistent with the same formula. The mismatch is upstream of these archived aggregate fields, not arithmetic after fields are written.

## 2. Why Golden Units Flip
- Golden units that remain selected in new score: `4`
- Golden units that flip positive in new score: `17`
- Flip categories: `{'protect_up_and_penalty_down': 7, 'penalty_down_only': 10}`

For score `protect - penalty`, a golden unit is lost when `protect` rises and/or `penalty=abs(proxy*cosine)` falls.
| unit | old score | new score | Δprotect | Δpenalty | old margin penalty-protect | new margin | main effect |
|---|---:|---:|---:|---:|---:|---:|---|
| `channel:31:7000` | -1.307e-03 | 8.100e-04 | 1.225e-04 | -1.995e-03 | 1.307e-03 | -8.100e-04 | `protect_up_and_penalty_down` |
| `channel:30:9255` | -9.789e-04 | 5.451e-04 | -9.375e-04 | -2.462e-03 | 9.789e-04 | -5.451e-04 | `penalty_down_only` |
| `channel:30:8030` | -6.706e-04 | 1.495e-04 | -8.275e-04 | -1.648e-03 | 6.706e-04 | -1.495e-04 | `penalty_down_only` |
| `channel:29:13691` | -6.324e-04 | 7.777e-04 | -5.279e-04 | -1.938e-03 | 6.324e-04 | -7.777e-04 | `penalty_down_only` |
| `channel:31:6059` | -2.835e-04 | 6.107e-05 | -4.592e-04 | -8.037e-04 | 2.835e-04 | -6.107e-05 | `penalty_down_only` |
| `channel:31:4751` | -2.198e-04 | 7.764e-04 | -1.001e-04 | -1.096e-03 | 2.198e-04 | -7.764e-04 | `penalty_down_only` |
| `channel:30:6959` | -2.022e-04 | 1.387e-03 | 6.711e-04 | -9.185e-04 | 2.022e-04 | -1.387e-03 | `protect_up_and_penalty_down` |
| `channel:31:683` | -1.914e-04 | 2.004e-04 | 2.676e-04 | -1.242e-04 | 1.914e-04 | -2.004e-04 | `protect_up_and_penalty_down` |
| `channel:31:3492` | -1.677e-04 | 7.618e-05 | -9.327e-04 | -1.177e-03 | 1.677e-04 | -7.618e-05 | `penalty_down_only` |
| `channel:30:9086` | -1.213e-04 | 2.696e-04 | -2.357e-04 | -6.266e-04 | 1.213e-04 | -2.696e-04 | `penalty_down_only` |
| `channel:30:6390` | -1.189e-04 | 8.361e-04 | -1.551e-04 | -1.110e-03 | 1.189e-04 | -8.361e-04 | `penalty_down_only` |
| `channel:30:5359` | -9.922e-05 | 5.425e-04 | -1.765e-04 | -8.182e-04 | 9.922e-05 | -5.425e-04 | `penalty_down_only` |
| `channel:31:13625` | -8.325e-05 | 2.707e-04 | 4.542e-05 | -3.086e-04 | 8.325e-05 | -2.707e-04 | `protect_up_and_penalty_down` |
| `channel:31:3820` | -3.734e-05 | 1.429e-04 | -2.623e-06 | -1.829e-04 | 3.734e-05 | -1.429e-04 | `penalty_down_only` |
| `channel:31:12721` | -2.997e-05 | 5.471e-04 | 2.677e-04 | -3.094e-04 | 2.997e-05 | -5.471e-04 | `protect_up_and_penalty_down` |
| `channel:30:538` | -2.871e-05 | 5.961e-04 | 4.554e-04 | -1.694e-04 | 2.871e-05 | -5.961e-04 | `protect_up_and_penalty_down` |
| `channel:31:7015` | -1.223e-05 | 2.551e-04 | 2.636e-04 | -3.770e-06 | 1.223e-05 | -2.551e-04 | `protect_up_and_penalty_down` |

## 3. Why New Extra Units Become Negative
- New selected units not in golden: `57`
- New-extra categories: `{'penalty_up_and_protect_down': 53, 'penalty_up_only': 3, 'protect_down_only': 1}`
| unit | old score | new score | Δprotect | Δpenalty | old margin | new margin | main effect |
|---|---:|---:|---:|---:|---:|---:|---|
| `channel:3:13690` | 3.460e-03 | -2.775e-03 | -2.038e-03 | 4.197e-03 | -3.460e-03 | 2.775e-03 | `penalty_up_and_protect_down` |
| `channel:31:7873` | 3.902e-03 | -1.483e-03 | -1.493e-03 | 3.893e-03 | -3.902e-03 | 1.483e-03 | `penalty_up_and_protect_down` |
| `channel:31:5311` | 1.955e-03 | -1.104e-03 | -1.864e-04 | 2.872e-03 | -1.955e-03 | 1.104e-03 | `penalty_up_and_protect_down` |
| `channel:30:1623` | 2.757e-03 | -8.793e-04 | -2.392e-03 | 1.245e-03 | -2.757e-03 | 8.793e-04 | `penalty_up_and_protect_down` |
| `channel:30:13628` | 1.564e-03 | -8.126e-04 | -8.825e-04 | 1.494e-03 | -1.564e-03 | 8.126e-04 | `penalty_up_and_protect_down` |
| `channel:30:7674` | 1.273e-03 | -7.668e-04 | -1.395e-03 | 6.441e-04 | -1.273e-03 | 7.668e-04 | `penalty_up_and_protect_down` |
| `channel:30:10006` | 3.094e-03 | -6.415e-04 | -8.521e-04 | 2.883e-03 | -3.094e-03 | 6.415e-04 | `penalty_up_and_protect_down` |
| `channel:25:4265` | 4.219e-03 | -6.128e-04 | -3.925e-03 | 9.062e-04 | -4.219e-03 | 6.128e-04 | `penalty_up_and_protect_down` |
| `channel:2:1683` | 2.487e-03 | -5.152e-04 | -1.076e-03 | 1.926e-03 | -2.487e-03 | 5.152e-04 | `penalty_up_and_protect_down` |
| `channel:29:10666` | 1.236e-02 | -4.923e-04 | -1.202e-02 | 8.237e-04 | -1.236e-02 | 4.923e-04 | `penalty_up_and_protect_down` |
| `channel:30:2951` | 4.893e-03 | -4.703e-04 | -3.928e-03 | 1.435e-03 | -4.893e-03 | 4.703e-04 | `penalty_up_and_protect_down` |
| `channel:19:3651` | 2.249e-03 | -4.665e-04 | -8.893e-04 | 1.826e-03 | -2.249e-03 | 4.665e-04 | `penalty_up_and_protect_down` |
| `channel:29:14165` | 8.966e-03 | -4.489e-04 | -8.770e-03 | 6.452e-04 | -8.966e-03 | 4.489e-04 | `penalty_up_and_protect_down` |
| `channel:28:106` | 2.786e-03 | -4.366e-04 | -2.352e-03 | 8.713e-04 | -2.786e-03 | 4.366e-04 | `penalty_up_and_protect_down` |
| `channel:28:13473` | 1.259e-03 | -4.255e-04 | -4.535e-04 | 1.231e-03 | -1.259e-03 | 4.255e-04 | `penalty_up_and_protect_down` |
| `channel:28:12074` | 1.455e-03 | -3.903e-04 | -1.220e-03 | 6.250e-04 | -1.455e-03 | 3.903e-04 | `penalty_up_and_protect_down` |
| `channel:31:884` | 2.121e-03 | -3.841e-04 | -1.670e-03 | 8.349e-04 | -2.121e-03 | 3.841e-04 | `penalty_up_and_protect_down` |
| `channel:31:4933` | 5.020e-04 | -3.786e-04 | 2.753e-04 | 1.156e-03 | -5.020e-04 | 3.786e-04 | `penalty_up_only` |
| `channel:31:3700` | 3.456e-04 | -3.502e-04 | -6.499e-04 | 4.582e-05 | -3.456e-04 | 3.502e-04 | `penalty_up_and_protect_down` |
| `channel:29:4856` | 5.525e-04 | -3.210e-04 | -1.144e-05 | 8.620e-04 | -5.525e-04 | 3.210e-04 | `penalty_up_and_protect_down` |
| `channel:31:2944` | 7.081e-04 | -2.911e-04 | -3.779e-05 | 9.614e-04 | -7.081e-04 | 2.911e-04 | `penalty_up_and_protect_down` |
| `channel:29:8993` | 1.863e-03 | -2.824e-04 | -3.333e-04 | 1.812e-03 | -1.863e-03 | 2.824e-04 | `penalty_up_and_protect_down` |
| `channel:31:422` | 2.820e-03 | -2.428e-04 | -1.634e-03 | 1.428e-03 | -2.820e-03 | 2.428e-04 | `penalty_up_and_protect_down` |
| `channel:15:10190` | 1.093e-03 | -2.348e-04 | -8.440e-05 | 1.243e-03 | -1.093e-03 | 2.348e-04 | `penalty_up_and_protect_down` |
| `channel:31:13823` | 1.305e-03 | -2.321e-04 | 1.863e-04 | 1.724e-03 | -1.305e-03 | 2.321e-04 | `penalty_up_only` |

## 4. Aggregate Deltas
### Golden 21 all
| field | mean | median | min | max |
|---|---:|---:|---:|---:|
| `delta_score` | 4.725e-04 | 3.918e-04 | -2.477e-03 | 2.117e-03 |
| `delta_protect` | -1.675e-04 | -1.001e-04 | -9.375e-04 | 6.711e-04 |
| `delta_penalty` | -6.400e-04 | -8.037e-04 | -2.462e-03 | 1.916e-03 |
| `delta_clean` | -7.674e-05 | -4.339e-05 | -6.506e-04 | 6.552e-04 |
| `delta_safe` | -1.815e-04 | -8.416e-05 | -7.983e-04 | 2.015e-04 |
| `delta_proxy` | 5.205e-04 | -3.395e-04 | -1.363e-02 | 1.371e-02 |
| `delta_cosine` | 1.065e-02 | -1.862e-02 | -2.307e-01 | 3.627e-01 |
### Golden 17 flipped positive
| field | mean | median | min | max |
|---|---:|---:|---:|---:|
| `delta_score` | 7.899e-04 | 6.248e-04 | 1.803e-04 | 2.117e-03 |
| `delta_protect` | -1.330e-04 | -1.001e-04 | -9.375e-04 | 6.711e-04 |
| `delta_penalty` | -9.229e-04 | -8.182e-04 | -2.462e-03 | -3.770e-06 |
| `delta_clean` | -5.549e-05 | -2.694e-05 | -6.506e-04 | 6.552e-04 |
| `delta_safe` | -1.551e-04 | -1.168e-05 | -7.983e-04 | 2.015e-04 |
| `delta_proxy` | -1.708e-03 | -1.495e-03 | -1.363e-02 | 9.127e-03 |
| `delta_cosine` | 3.413e-02 | 3.268e-02 | -2.307e-01 | 3.627e-01 |
### New 57 extras
| field | mean | median | min | max |
|---|---:|---:|---:|---:|
| `delta_score` | -2.545e-03 | -1.658e-03 | -1.285e-02 | -3.283e-04 |
| `delta_protect` | -1.462e-03 | -8.361e-04 | -1.202e-02 | 2.753e-04 |
| `delta_penalty` | 1.083e-03 | 7.018e-04 | -3.812e-04 | 9.161e-03 |
| `delta_clean` | -9.699e-04 | -4.599e-04 | -8.126e-03 | 2.227e-04 |
| `delta_safe` | -9.852e-04 | -6.126e-04 | -7.798e-03 | 1.051e-04 |
| `delta_proxy` | 5.013e-03 | 2.824e-03 | -8.593e-04 | 4.619e-02 |
| `delta_cosine` | -4.167e-02 | -8.057e-03 | -4.874e-01 | 3.179e-01 |

## 5. Near-Zero Sensitivity
| eps around zero | old count | new count | overlap |
|---:|---:|---:|---:|
| `1e-05` | 0 | 9 | 0 |
| `5e-05` | 8 | 39 | 0 |
| `0.0001` | 19 | 80 | 1 |
| `0.0005` | 49518 | 114661 | 38700 |
| `0.001` | 318433 | 390412 | 306461 |

## Interpretation
- The formula arithmetic itself checks out for both old and new files.
- The loss of golden units is not caused by a single universal component: for flipped golden units, both `protect` increases and `penalty` decreases appear, often together.
- New extra units usually become selected because their new penalty increases and/or protect decreases relative to old.
- Therefore, the divergence occurs before the aggregate `unit_scores.json` fields are written: in the generation of clean/safe/proxy gradients and cosine directions. With the artifacts currently available, this is the earliest observable divergence point.
