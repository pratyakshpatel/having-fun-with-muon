#!/usr/bin/env bash
set -euo pipefail

if [[ ! -f data/tokenized/fineweb_edu_10bt/dataset_manifest.json ]]; then
  python train_muon_atlas.py --prepare_dataset fineweb_edu_10bt --target_tokens "${PREP_TOKENS:-100000000}"
fi

MODEL_SIZE="${MODEL_SIZE:-small_35m}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-512}"
MICRO_BATCH="${BATCH_SIZE:-8}"
BEST_ROUTED_MUON="${BEST_ROUTED_MUON:-muon_no_qk}"

for effective_tokens in 64000 128000 256000 512000; do
  grad_accum=$(( (effective_tokens + MICRO_BATCH * CONTEXT_LENGTH - 1) / (MICRO_BATCH * CONTEXT_LENGTH) ))
  for routing in adamw_all muon_all_hidden "$BEST_ROUTED_MUON"; do
    python train_muon_atlas.py \
      --dataset fineweb_edu_10bt \
      --model_size "$MODEL_SIZE" \
      --routing "$routing" \
      --seed 0 \
      --max_tokens "${MAX_TOKENS:-100000000}" \
      --batch_size "$MICRO_BATCH" \
      --grad_accum "$grad_accum" \
      --context_length "$CONTEXT_LENGTH" \
      --eval_interval "${EVAL_INTERVAL:-1000}" \
      --geometry_interval "${GEOMETRY_INTERVAL:-5000}" \
      --sample_interval "${SAMPLE_INTERVAL:-5000}" \
      --save_every "${SAVE_EVERY:-5000}" \
      --dtype "${DTYPE:-bf16}" \
      --compile "${COMPILE:-false}" \
      --out_dir runs
  done
done
