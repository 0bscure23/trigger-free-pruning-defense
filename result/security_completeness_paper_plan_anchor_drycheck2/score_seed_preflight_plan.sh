#!/usr/bin/env bash
set -euo pipefail
RUN=1 SCORE_SEED=13 PROMPT_TEMPLATE=chat SCORE_MAX_LENGTH=256 OUT_ROOT=/home/lizhy/plp/trigger-free-pruning-defense-round2/result/security_completeness_paper_plan_anchor_drycheck2/score_seed_preflight /home/lizhy/plp/TRANSFER/run_llama_word_score_template_variant.sh
RUN=1 SCORE_SEED=17 PROMPT_TEMPLATE=chat SCORE_MAX_LENGTH=256 OUT_ROOT=/home/lizhy/plp/trigger-free-pruning-defense-round2/result/security_completeness_paper_plan_anchor_drycheck2/score_seed_preflight /home/lizhy/plp/TRANSFER/run_llama_word_score_template_variant.sh
RUN=1 SCORE_SEED=23 PROMPT_TEMPLATE=chat SCORE_MAX_LENGTH=256 OUT_ROOT=/home/lizhy/plp/trigger-free-pruning-defense-round2/result/security_completeness_paper_plan_anchor_drycheck2/score_seed_preflight /home/lizhy/plp/TRANSFER/run_llama_word_score_template_variant.sh
