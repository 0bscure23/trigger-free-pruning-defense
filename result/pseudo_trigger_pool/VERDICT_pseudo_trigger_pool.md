# Pseudo-Trigger Pool Probe

Date: 2026-06-22

## Setup

- No known real trigger was used.
- Candidate pool: generic suffixes only, including chat format tokens, roleplay phrases,
  debug/mode phrases, delimiters, rare synthetic phrases, and style markers.
- Candidate screening: 59 candidates on 8 harmful-no-trigger + 8 benign prompts.
- Scoring: top screened pseudo suffixes, 4 clean scoring pairs, 512-unit plan.

Top screened candidates were mostly format tokens:

1. `ASSISTANT:`
2. `BEGIN RESPONSE`
3. `</assistant>`
4. `USER:`
5. `<assistant>`

## Result

| Plan | ASR | HR | BFR | Rolling PPL |
|---|---:|---:|---:|---:|
| pseudo-top512 ablate-only | 0.9000 | 0.400 | 0.150 | 11.97 |

Overlap:

- trigger-aware oracle: 2 / 512, J=0.001
- FGSM baseline: 110 / 512, J=0.120
- trigger-free causal `eps=0.01`: 2 / 512, J=0.001

Layer distribution is highly degenerate: 377 / 512 units are in layer 0.

## Verdict

Failed. The generic pseudo-trigger pool mostly discovers chat-format/template sensitivity,
not a backdoor-relevant circuit. It preserves PPL, but ASR remains around raw-model level.
This should not be recovered.

Next action: test clean/reference-model difference if a compatible clean checkpoint exists locally.
