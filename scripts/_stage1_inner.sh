#!/usr/bin/env bash
set -euo pipefail

for routing in adamw_all muon_all_hidden muon_mlp_only muon_attn_only muon_vo_only muon_qk_only muon_no_qk; do
  for seed in 0 1; do
    python train_muon_atlas.py \
      --dataset tinystories \
      --model_size tiny_12m \
      --routing "$routing" \
      --seed "$seed" \
      --max_tokens "${MAX_TOKENS:-50000000}" \
      --batch_size "${BATCH_SIZE:-32}" \
      --grad_accum "${GRAD_ACCUM:-16}" \
      --context_length "${CONTEXT_LENGTH:-256}" \
      --eval_interval "${EVAL_INTERVAL:-500}" \
      --geometry_interval "${GEOMETRY_INTERVAL:-2000}" \
      --sample_interval "${SAMPLE_INTERVAL:-2000}" \
      --save_every "${SAVE_EVERY:-2000}" \
      --dtype "${DTYPE:-bf16}" \
      --compile "${COMPILE:-false}" \
      --out_dir runs
  done
done
