#!/usr/bin/env bash
set -euo pipefail

DATASET="${DATASET:-tinystories}"
MODEL_SIZE="${MODEL_SIZE:-small_35m}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-512}"

if [[ "$DATASET" == "fineweb_edu_10bt" && ! -f data/tokenized/fineweb_edu_10bt/dataset_manifest.json ]]; then
  python train_muon_atlas.py --prepare_dataset fineweb_edu_10bt --target_tokens "${PREP_TOKENS:-100000000}"
fi

for routing in muon_early_layers muon_middle_layers muon_late_layers muon_late_mlp_only muon_late_vo_only; do
  for seed in ${SEEDS:-0}; do
    python train_muon_atlas.py \
      --dataset "$DATASET" \
      --model_size "$MODEL_SIZE" \
      --routing "$routing" \
      --seed "$seed" \
      --max_tokens "${MAX_TOKENS:-50000000}" \
      --batch_size "${BATCH_SIZE:-16}" \
      --grad_accum "${GRAD_ACCUM:-16}" \
      --context_length "$CONTEXT_LENGTH" \
      --eval_interval "${EVAL_INTERVAL:-500}" \
      --geometry_interval "${GEOMETRY_INTERVAL:-2000}" \
      --sample_interval "${SAMPLE_INTERVAL:-2000}" \
      --save_every "${SAVE_EVERY:-2000}" \
      --dtype "${DTYPE:-bf16}" \
      --compile "${COMPILE:-false}" \
      --out_dir runs
  done
done
