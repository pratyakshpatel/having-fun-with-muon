# Muon Routing Atlas

**Which Transformer Parameters Benefit from Orthogonalized Optimization?**

This repo is a single-GPU research harness for testing whether Muon's gains in GPT-style pretraining come from applying it to every hidden matrix or mainly from specific transformer modules such as FFN, V/O attention projections, Q/K projections, or depth regions.

The code trains compact decoder-only language models, implements AdamW and Muon+AdamW hybrid optimization, logs optimizer routing and update geometry, generates qualitative samples, builds plots/tables, and compiles a LaTeX report.

## Research Question

I treat optimizer assignment as the experimental variable:

- AdamW everywhere
- Muon on all hidden 2D matrices
- Muon only on MLP matrices
- Muon only on attention matrices
- Muon only on V/O matrices
- Muon only on Q/K matrices
- Muon on everything except Q/K
- Muon by early/middle/late layers

The hypothesis is that FFN and V/O matrices may recover much of Muon's gain without applying Muon to every hidden matrix.

## Source Links

- TinyStories dataset: <https://huggingface.co/datasets/roneneldan/TinyStories>
- TinyStories train text: <https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-train.txt>
- TinyStories valid text: <https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-valid.txt>
- TinyStories paper: <https://arxiv.org/abs/2305.07759>
- FineWeb-Edu dataset: <https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu>
- FineWeb-Edu sample-10BT tree: <https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/tree/main/sample/10BT>
- FineWeb paper: <https://arxiv.org/abs/2406.17557>
- Muon reference implementation: <https://github.com/KellerJordan/Muon>
- Muon writeup: <https://kellerjordan.github.io/posts/muon/>
- Hugging Face streaming docs: <https://huggingface.co/docs/datasets/en/stream>

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

The tmux scripts also create `.venv` and install requirements automatically if `.venv` is missing.

## Data

Dataset files are not committed to this repository. Local raw downloads live under
`data/raw/`, tokenized binary shards live under `data/tokenized/`, and both paths
are ignored by git except for `.gitkeep` placeholders.

Prepare TinyStories with the repository helper:

```bash
python train_muon_atlas.py --prepare_dataset tinystories
```

The helper downloads these Hugging Face files when they are missing:

- <https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-train.txt>
- <https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-valid.txt>

You can also fetch them manually:

```bash
mkdir -p data/raw/tinystories
curl -L -o data/raw/tinystories/TinyStories-train.txt https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-train.txt
curl -L -o data/raw/tinystories/TinyStories-valid.txt https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-valid.txt
python train_muon_atlas.py --prepare_dataset tinystories
```

Prepare FineWeb-Edu sample-10BT through Hugging Face streaming:

```bash
python train_muon_atlas.py --prepare_dataset fineweb_edu_10bt --target_tokens 1000000000
```

Use a smaller `--target_tokens` value for a quick local test. The FineWeb-Edu
source is <https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/tree/main/sample/10BT>.

## Smoke Test

```bash
bash scripts/tmux_smoke.sh
tmux attach -t muon_smoke
```

The smoke test runs TinyStories with the smoke model for:

- `adamw_all`
- `muon_all_hidden`
- `muon_no_qk`

It also runs analysis/report asset generation.

## Stage 1: TinyStories Routing Sweep

```bash
bash scripts/tmux_stage1_tinystories.sh
tmux attach -t muon_stage1
```

Override token budget or microbatch settings with environment variables:

```bash
MAX_TOKENS=10000000 BATCH_SIZE=16 GRAD_ACCUM=16 bash scripts/tmux_stage1_tinystories.sh
```

## Stage 2: FineWeb-Edu Comparison

```bash
bash scripts/tmux_stage2_finewebedu.sh
tmux attach -t muon_stage2
```

The stage script prepares FineWeb-Edu sample-10BT shards first if they do not exist:

```bash
python train_muon_atlas.py --prepare_dataset fineweb_edu_10bt --target_tokens 1000000000
```

Use `MAX_TOKENS`, `PREP_TOKENS`, and `CONTEXT_LENGTH` to scale the run.

## Stage 3 and Stage 4

```bash
bash scripts/tmux_stage3_layerwise.sh
bash scripts/tmux_stage4_batchsize.sh
```

Stage 3 runs layerwise routing modes. Stage 4 runs effective batch-size stress tests with AdamW, full-hidden Muon, and `BEST_ROUTED_MUON` which defaults to `muon_no_qk`.

## Analysis

```bash
bash scripts/tmux_analyze.sh
```

This aggregates runs under `runs/` and writes:

- `report/figures/validation_loss_vs_tokens_tinystories.png`
- `report/figures/validation_loss_vs_tokens_finewebedu.png`
- `report/figures/validation_loss_vs_wallclock.png`
- `report/figures/final_loss_barplot.png`
- `report/figures/routing_heatmap.png`
- `report/figures/layerwise_heatmap.png`
- `report/figures/update_effective_rank_by_module.png`
- `report/figures/update_norm_by_module.png`
- `report/figures/singular_value_spectrum_examples.png`
- `report/figures/qk_logit_growth.png`
- `report/figures/batch_size_scaling.png`
- `report/tables/*.tex`

Missing runs produce placeholder figures/tables rather than crashing.

## Report

```bash
bash scripts/tmux_report.sh
```

The report script runs analysis, generates LaTeX assets, then tries:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

If `latexmk` is unavailable, it tries:

```bash
tectonic main.tex
```

If neither exists, it still writes `report/main.tex` and creates `report/compiled/PDF_NOT_BUILT.txt` with this message:

```text
PDF compilation skipped because latexmk/tectonic was not found.
```

## Monitoring

```bash
bash scripts/monitor.sh
```

This shows `tmux ls`, `nvidia-smi`, and the last 40 lines of `logs/*.log`.

## Expected Output Files

Each run creates:

- `runs/{timestamp}_{dataset}_{model_size}_{routing}_seed{seed}/config_resolved.yaml`
- `runs/{timestamp}_{dataset}_{model_size}_{routing}_seed{seed}/metrics.csv`
- `runs/{timestamp}_{dataset}_{model_size}_{routing}_seed{seed}/optimizer_groups.json`
- `runs/{timestamp}_{dataset}_{model_size}_{routing}_seed{seed}/samples.jsonl`
- `runs/{timestamp}_{dataset}_{model_size}_{routing}_seed{seed}/geometry_metrics.csv`
- `runs/{timestamp}_{dataset}_{model_size}_{routing}_seed{seed}/checkpoint_last.pt`
- `runs/{timestamp}_{dataset}_{model_size}_{routing}_seed{seed}/checkpoint_best.pt`

Report outputs include:

- `report/figures/*.png`
- `report/tables/*.tex`
- `report/main.tex`
- `report/compiled/muon_routing_atlas_report.pdf` when a TeX engine is installed

## Reproducibility Notes

- GPT-2 BPE tokenization is used through `tiktoken`.
- TinyStories is saved under `data/tokenized/tinystories/`.
- FineWeb-Edu shards are saved under `data/tokenized/fineweb_edu_10bt/`.
- Each prepared dataset writes `dataset_manifest.json`.
- Each run saves random seeds, resolved config, optimizer groups, and git commit hash when available.
- Training refuses to use the same file for train and validation data.
- Muon routing is checked so embeddings, heads, norms, and biases do not enter the Muon group.

## Known Limitations

The default experiments are intentionally small enough for a single GPU. The models are small, seeds are limited, token budgets are short relative to full pretraining, and TinyStories is synthetic. The default hyperparameter budget is controlled and small, not a global optimizer search.
