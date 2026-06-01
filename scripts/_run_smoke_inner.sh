#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs runs

for routing in adamw_all muon_all_hidden muon_no_qk; do
  python train_muon_atlas.py \
    --dataset tinystories \
    --model_size smoke \
    --routing "$routing" \
    --seed 0 \
    --max_tokens "${MAX_TOKENS:-1000000}" \
    --batch_size "${BATCH_SIZE:-8}" \
    --grad_accum "${GRAD_ACCUM:-2}" \
    --context_length 128 \
    --eval_interval 50 \
    --geometry_interval 100 \
    --sample_interval 100 \
    --save_every 100 \
    --dtype bf16 \
    --compile false \
    --out_dir runs
done

python analyze_muon_atlas.py --runs_dir runs --report_dir report
python generate_report_assets.py --runs_dir runs --report_dir report
