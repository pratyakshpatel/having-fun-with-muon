#!/usr/bin/env bash
set -euo pipefail

PREP_TOKENS="${PREP_TOKENS:-${MAX_TOKENS:-200000000}}"
if [[ ! -f data/tokenized/fineweb_edu_10bt/dataset_manifest.json ]]; then
  python train_muon_atlas.py --prepare_dataset fineweb_edu_10bt --target_tokens "$PREP_TOKENS"
fi

for routing in adamw_all muon_all_hidden muon_mlp_only muon_vo_only muon_no_qk; do
  for seed in 0 1; do
    python train_muon_atlas.py \
      --dataset fineweb_edu_10bt \
      --model_size mid_85m \
      --routing "$routing" \
      --seed "$seed" \
      --max_tokens "${MAX_TOKENS:-200000000}" \
      --batch_size "${BATCH_SIZE:-8}" \
      --grad_accum "${GRAD_ACCUM:-16}" \
      --context_length "${CONTEXT_LENGTH:-512}" \
      --eval_interval "${EVAL_INTERVAL:-1000}" \
      --geometry_interval "${GEOMETRY_INTERVAL:-5000}" \
      --sample_interval "${SAMPLE_INTERVAL:-5000}" \
      --save_every "${SAVE_EVERY:-5000}" \
      --dtype "${DTYPE:-bf16}" \
      --compile "${COMPILE:-false}" \
      --out_dir runs
  done
done
