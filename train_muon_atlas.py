#!/usr/bin/env python3
"""Train GPT-style models for the Muon Routing Atlas experiments."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from tqdm import tqdm

from muon_impl import Muon


ROUTINGS = [
    "adamw_all",
    "muon_all_hidden",
    "muon_mlp_only",
    "muon_attn_only",
    "muon_vo_only",
    "muon_qk_only",
    "muon_no_qk",
    "muon_early_layers",
    "muon_middle_layers",
    "muon_late_layers",
    "muon_late_mlp_only",
    "muon_late_vo_only",
]

MODEL_SIZES = {
    "smoke": dict(n_layer=2, n_embd=128, n_head=4),
    "tiny_12m": dict(n_layer=6, n_embd=384, n_head=6),
    "small_35m": dict(n_layer=8, n_embd=512, n_head=8),
    "mid_85m": dict(n_layer=12, n_embd=768, n_head=12),
}

TINY_PROMPTS = [
    "Once upon a time, there was a little robot",
    "Lily found a strange blue stone in the garden",
    "The puppy wanted to learn how to fly",
]

FINEWEB_PROMPTS = [
    "In simple terms, photosynthesis is",
    "The history of the internet began when",
    "A useful way to understand gravity is",
    "Machine learning models are trained by",
]


@dataclass
class GPTConfig:
    vocab_size: int = 50304
    block_size: int = 256
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.0


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.q_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.k_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.v_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.o_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.dropout = cfg.dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq, dim = x.shape
        q = self.q_proj(x).view(bsz, seq, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq, self.n_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq, self.n_head, self.head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0.0, is_causal=True
        )
        y = y.transpose(1, 2).contiguous().view(bsz, seq, dim)
        return self.o_proj(y)


class SwiGLU(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        inner = int(8 * cfg.n_embd / 3)
        inner = ((inner + 255) // 256) * 256
        self.gate_proj = nn.Linear(cfg.n_embd, inner, bias=False)
        self.up_proj = nn.Linear(cfg.n_embd, inner, bias=False)
        self.down_proj = nn.Linear(inner, cfg.n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.norm1 = RMSNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.norm2 = RMSNorm(cfg.n_embd)
        self.mlp = SwiGLU(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.position_embedding = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.final_norm = RMSNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        bsz, seq = idx.shape
        if seq > self.cfg.block_size:
            raise ValueError(f"sequence length {seq} exceeds block size {self.cfg.block_size}")
        pos = torch.arange(0, seq, dtype=torch.long, device=idx.device)
        x = self.token_embedding(idx) + self.position_embedding(pos)[None, :, :]
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.final_norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.8,
        top_k: int = 50,
    ) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k > 0:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_id), dim=1)
        return idx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--prepare_dataset", choices=["tinystories", "fineweb_edu_10bt"])
    parser.add_argument("--target_tokens", type=int, default=1_000_000_000)
    parser.add_argument("--shard_tokens", type=int, default=50_000_000)
    parser.add_argument("--dataset", choices=["tinystories", "fineweb_edu_10bt"], default="tinystories")
    parser.add_argument("--model_size", choices=list(MODEL_SIZES), default="tiny_12m")
    parser.add_argument("--routing", choices=ROUTINGS, default="adamw_all")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_tokens", type=int, default=50_000_000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--grad_accum", type=int, default=16)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--eval_interval", type=int, default=500)
    parser.add_argument("--eval_batches", type=int, default=20)
    parser.add_argument("--geometry_interval", type=int, default=2000)
    parser.add_argument("--sample_interval", type=int, default=2000)
    parser.add_argument("--save_every", type=int, default=2000)
    parser.add_argument("--out_dir", type=str, default="runs")
    parser.add_argument("--tokenizer", choices=["gpt2"], default="gpt2")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--compile", type=str2bool, default=False)
    parser.add_argument("--resume", type=str)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--wandb", type=str2bool, default=False)
    parser.add_argument("--adamw_lr", type=float, default=3e-4)
    parser.add_argument("--adamw_weight_decay", type=float, default=0.1)
    parser.add_argument("--adamw_beta1", type=float, default=0.9)
    parser.add_argument("--adamw_beta2", type=float, default=0.95)
    parser.add_argument("--muon_lr", type=float, default=0.02)
    parser.add_argument("--muon_weight_decay", type=float, default=0.05)
    parser.add_argument("--muon_momentum", type=float, default=0.95)
    args = parser.parse_args()
    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f) or {}
        for key, value in cfg.items():
            if hasattr(args, key) and getattr(args, key) == parser.get_default(key):
                setattr(args, key, value)
    return args


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return value.lower() in {"1", "true", "yes", "y", "on"}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def get_git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def get_tokenizer():
    import tiktoken

    return tiktoken.get_encoding("gpt2")


def write_bin(path: Path, tokens: Iterable[int]) -> int:
    arr = np.fromiter(tokens, dtype=np.uint32)
    if arr.size == 0:
        raise ValueError(f"refusing to write empty token file: {path}")
    if arr.max(initial=0) > np.iinfo(np.uint16).max:
        raise ValueError("token ids exceed uint16 range")
    path.parent.mkdir(parents=True, exist_ok=True)
    arr.astype(np.uint16).tofile(path)
    return int(arr.size)


def prepare_tinystories(root: Path) -> None:
    raw = root / "data/raw/tinystories"
    out = root / "data/tokenized/tinystories"
    train_bin, valid_bin = out / "train.bin", out / "valid.bin"
    if train_bin.exists() and valid_bin.exists():
        print(f"TinyStories tokenized files already exist in {out}")
        return

    raw.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    train_txt, valid_txt = raw / "TinyStories-train.txt", raw / "TinyStories-valid.txt"
    try:
        from huggingface_hub import hf_hub_download

        for filename, target in [
            ("TinyStories-train.txt", train_txt),
            ("TinyStories-valid.txt", valid_txt),
        ]:
            cached = hf_hub_download("roneneldan/TinyStories", filename=filename, repo_type="dataset")
            shutil.copyfile(cached, target)
    except Exception as exc:
        print(f"Direct TinyStories download failed, falling back to datasets.load_dataset: {exc}")
        from datasets import load_dataset

        ds = load_dataset("roneneldan/TinyStories")
        train_txt.write_text("\n".join(ds["train"]["text"]), encoding="utf-8")
        valid_split = ds["validation"] if "validation" in ds else ds["train"].select(range(10000))
        valid_txt.write_text("\n".join(valid_split["text"]), encoding="utf-8")

    enc = get_tokenizer()
    eot = enc.eot_token
    for src, dst in [(train_txt, train_bin), (valid_txt, valid_bin)]:
        total = write_bin(dst, encode_text_file(src, enc, eot))
        print(f"Wrote {total:,} tokens to {dst}")

    manifest = {
        "dataset": "tinystories",
        "source_url": "https://huggingface.co/datasets/roneneldan/TinyStories",
        "tokenizer": "gpt2",
        "train_path": str(train_bin),
        "valid_path": str(valid_bin),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    (out / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def encode_text_file(path: Path, enc, eot: int):
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in tqdm(f, desc=f"tokenizing {path.name}"):
            text = line.strip()
            if not text:
                continue
            yield from enc.encode_ordinary(text)
            yield eot


def prepare_fineweb(root: Path, target_tokens: int, shard_tokens: int, script_args: dict) -> None:
    from datasets import load_dataset

    out = root / "data/tokenized/fineweb_edu_10bt"
    out.mkdir(parents=True, exist_ok=True)
    enc = get_tokenizer()
    eot = enc.eot_token
    valid_target = min(1_000_000, max(100_000, target_tokens // 100))
    valid_tokens: list[int] = []
    shard: list[int] = []
    shard_paths: list[str] = []
    train_tokens = 0
    shard_idx = 0

    ds = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True)
    pbar = tqdm(total=target_tokens, desc="preparing FineWeb-Edu tokens")
    for row in ds:
        toks = enc.encode_ordinary(row.get("text", "")) + [eot]
        for tok in toks:
            if len(valid_tokens) < valid_target:
                valid_tokens.append(tok)
                continue
            shard.append(tok)
            train_tokens += 1
            pbar.update(1)
            if len(shard) >= shard_tokens:
                path = out / f"train_{shard_idx:06d}.bin"
                write_bin(path, shard)
                shard_paths.append(str(path))
                shard.clear()
                shard_idx += 1
            if train_tokens >= target_tokens:
                break
        if train_tokens >= target_tokens:
            break
    pbar.close()
    if shard:
        path = out / f"train_{shard_idx:06d}.bin"
        write_bin(path, shard)
        shard_paths.append(str(path))
    valid_path = out / "valid.bin"
    write_bin(valid_path, valid_tokens)
    if str(valid_path) in shard_paths:
        raise RuntimeError("FineWeb validation file is also listed as a train shard")
    manifest = {
        "dataset": "fineweb_edu_10bt",
        "source_url": "https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/tree/main/sample/10BT",
        "tokenizer": "gpt2",
        "number_of_tokens_prepared": train_tokens,
        "validation_tokens": len(valid_tokens),
        "shard_paths": shard_paths,
        "valid_path": str(valid_path),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "script_args": script_args,
    }
    (out / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Prepared {train_tokens:,} train tokens and {len(valid_tokens):,} validation tokens in {out}")


class BinaryTokenDataset:
    def __init__(self, paths: list[Path], block_size: int, device: str) -> None:
        self.paths = paths
        self.block_size = block_size
        self.device = device
        self.arrays = [np.memmap(path, dtype=np.uint16, mode="r") for path in paths]
        usable = [max(0, len(arr) - block_size - 1) for arr in self.arrays]
        if sum(usable) <= 0:
            raise ValueError(f"not enough tokens for block_size={block_size}: {paths}")
        self.weights = np.asarray(usable, dtype=np.float64) / sum(usable)

    def get_batch(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        xs, ys = [], []
        choices = np.random.choice(len(self.arrays), size=batch_size, p=self.weights)
        for idx in choices:
            arr = self.arrays[int(idx)]
            start = np.random.randint(0, len(arr) - self.block_size - 1)
            chunk = np.asarray(arr[start : start + self.block_size + 1], dtype=np.int64)
            xs.append(torch.from_numpy(chunk[:-1].copy()))
            ys.append(torch.from_numpy(chunk[1:].copy()))
        x = torch.stack(xs).to(self.device, non_blocking=True)
        y = torch.stack(ys).to(self.device, non_blocking=True)
        return x, y


def build_datasets(root: Path, dataset: str, block_size: int, device: str):
    if dataset == "tinystories":
        prepare_tinystories(root)
        base = root / "data/tokenized/tinystories"
        train_paths = [base / "train.bin"]
        valid_paths = [base / "valid.bin"]
    else:
        base = root / "data/tokenized/fineweb_edu_10bt"
        manifest_path = base / "dataset_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                "FineWeb-Edu token shards not found. Run: "
                "python train_muon_atlas.py --prepare_dataset fineweb_edu_10bt --target_tokens N"
            )
        manifest = json.loads(manifest_path.read_text())
        train_paths = [Path(p) for p in manifest["shard_paths"]]
        valid_paths = [Path(manifest["valid_path"])]

    if set(map(str, train_paths)) & set(map(str, valid_paths)):
        raise RuntimeError("train and validation data point to the same file")
    return BinaryTokenDataset(train_paths, block_size, device), BinaryTokenDataset(valid_paths, block_size, device)


def is_hidden_matrix(name: str, param: nn.Parameter) -> bool:
    if param.ndim != 2:
        return False
    if "token_embedding" in name or "position_embedding" in name or "lm_head" in name:
        return False
    if ".norm" in name or name.endswith(".bias"):
        return False
    return any(
        token in name
        for token in [
            ".attn.q_proj.weight",
            ".attn.k_proj.weight",
            ".attn.v_proj.weight",
            ".attn.o_proj.weight",
            ".mlp.gate_proj.weight",
            ".mlp.up_proj.weight",
            ".mlp.down_proj.weight",
        ]
    )


def block_index(name: str) -> int | None:
    parts = name.split(".")
    if len(parts) > 2 and parts[0] == "blocks":
        return int(parts[1])
    return None


def in_layer_third(name: str, n_layer: int, third: str) -> bool:
    idx = block_index(name)
    if idx is None:
        return False
    first_end = n_layer // 3
    second_end = 2 * n_layer // 3
    if third == "early":
        return idx < first_end
    if third == "middle":
        return first_end <= idx < second_end
    if third == "late":
        return idx >= second_end
    raise ValueError(third)


def selected_for_muon(name: str, param: nn.Parameter, routing: str, n_layer: int) -> bool:
    if routing == "adamw_all":
        return False
    if not is_hidden_matrix(name, param):
        return False
    is_mlp = ".mlp." in name
    is_attn = ".attn." in name
    is_vo = ".attn.v_proj.weight" in name or ".attn.o_proj.weight" in name
    is_qk = ".attn.q_proj.weight" in name or ".attn.k_proj.weight" in name
    if routing == "muon_all_hidden":
        return True
    if routing == "muon_mlp_only":
        return is_mlp
    if routing == "muon_attn_only":
        return is_attn
    if routing == "muon_vo_only":
        return is_vo
    if routing == "muon_qk_only":
        return is_qk
    if routing == "muon_no_qk":
        return is_mlp or is_vo
    if routing == "muon_early_layers":
        return in_layer_third(name, n_layer, "early")
    if routing == "muon_middle_layers":
        return in_layer_third(name, n_layer, "middle")
    if routing == "muon_late_layers":
        return in_layer_third(name, n_layer, "late")
    if routing == "muon_late_mlp_only":
        return in_layer_third(name, n_layer, "late") and is_mlp
    if routing == "muon_late_vo_only":
        return in_layer_third(name, n_layer, "late") and is_vo
    raise ValueError(routing)


def build_optimizers(model: GPT, args: argparse.Namespace, run_dir: Path):
    named = list(model.named_parameters())
    muon_named, adamw_named = [], []
    for name, p in named:
        if not p.requires_grad:
            continue
        if selected_for_muon(name, p, args.routing, model.cfg.n_layer):
            muon_named.append((name, p))
        else:
            adamw_named.append((name, p))

    if args.routing == "adamw_all" and muon_named:
        raise RuntimeError("adamw_all routing produced a non-empty Muon group")
    for name, p in muon_named:
        if p.ndim != 2:
            raise RuntimeError(f"Muon parameter is not 2D: {name}")
        if any(bad in name for bad in ["token_embedding", "position_embedding", "lm_head", "norm"]):
            raise RuntimeError(f"forbidden parameter routed to Muon: {name}")
    if args.routing != "adamw_all" and not muon_named:
        print(f"WARNING: routing {args.routing} matched no Muon parameters")

    muon_count = sum(p.numel() for _, p in muon_named)
    adamw_count = sum(p.numel() for _, p in adamw_named)
    total_count = muon_count + adamw_count
    groups = {"routing": args.routing, "muon": [n for n, _ in muon_named], "adamw": [n for n, _ in adamw_named]}
    (run_dir / "optimizer_groups.json").write_text(json.dumps(groups, indent=2), encoding="utf-8")
    print(
        f"Optimizer routing: {len(muon_named)} Muon tensors, {len(adamw_named)} AdamW tensors, "
        f"{muon_count:,} Muon params, {adamw_count:,} AdamW params, "
        f"{100.0 * muon_count / max(1, total_count):.2f}% Muon"
    )

    adamw = torch.optim.AdamW(
        [p for _, p in adamw_named],
        lr=args.adamw_lr,
        betas=(args.adamw_beta1, args.adamw_beta2),
        weight_decay=args.adamw_weight_decay,
    )
    muon = None
    if muon_named:
        muon = Muon(
            [p for _, p in muon_named],
            lr=args.muon_lr,
            momentum=args.muon_momentum,
            weight_decay=args.muon_weight_decay,
        )
    return adamw, muon, groups


def get_amp_dtype(requested: str, device: str):
    if device.startswith("cuda"):
        if requested == "bf16" and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        if requested == "bf16":
            print("WARNING: bf16 unavailable; falling back to fp16.")
            return torch.float16
        if requested == "fp16":
            return torch.float16
    if requested != "fp32":
        print("WARNING: requested mixed precision on non-CUDA device; using fp32.")
    return torch.float32


def cosine_lr(base_lr: float, step: int, max_steps: int, warmup_steps: int) -> float:
    if step < warmup_steps:
        return base_lr * (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return 0.1 * base_lr + 0.9 * base_lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def set_lr(optimizer, lr: float) -> None:
    if optimizer is None:
        return
    for group in optimizer.param_groups:
        group["lr"] = lr


@torch.no_grad()
def estimate_loss(model: GPT, train_data: BinaryTokenDataset, valid_data: BinaryTokenDataset, args, amp_dtype) -> tuple[float, float]:
    model.eval()
    losses = []
    for split, data in [("train", train_data), ("valid", valid_data)]:
        split_losses = []
        for _ in range(args.eval_batches):
            x, y = data.get_batch(args.batch_size)
            with torch.autocast(device_type=args.device.split(":")[0], dtype=amp_dtype, enabled=amp_dtype != torch.float32):
                _, loss = model(x, y)
            split_losses.append(float(loss.item()))
        losses.append(float(np.mean(split_losses)))
    model.train()
    return losses[0], losses[1]


def orthogonality_error(matrix: torch.Tensor) -> float:
    mat = matrix.detach().float()
    if mat.ndim != 2:
        return float("nan")
    if mat.shape[0] <= mat.shape[1]:
        gram = mat @ mat.T
        ident = torch.eye(mat.shape[0], device=mat.device)
    else:
        gram = mat.T @ mat
        ident = torch.eye(mat.shape[1], device=mat.device)
    return float((gram - ident).norm() / (ident.norm() + 1e-12))


@torch.no_grad()
def collect_geometry(model: GPT, step: int, tokens_seen: int, routing: str) -> list[dict]:
    rows = []
    for name, p in model.named_parameters():
        if not is_hidden_matrix(name, p):
            continue
        grad_norm = float(p.grad.norm().item()) if p.grad is not None else 0.0
        update = getattr(p, "_muon_last_update", None)
        update_norm = float(update.norm().item()) if update is not None else 0.0
        weight = p.detach().float()
        weight_norm = float(weight.norm().item())
        try:
            sv = torch.linalg.svdvals(weight.cpu())
            total = sv.sum().item()
            probs = sv / max(total, 1e-12)
            effective_rank = float(torch.exp(-(probs * torch.log(probs + 1e-12)).sum()).item())
            top_sv = float(sv[0].item())
        except Exception:
            effective_rank = float("nan")
            top_sv = float("nan")
        rows.append(
            {
                "step": step,
                "tokens_seen": tokens_seen,
                "module_name": ".".join(name.split(".")[2:4]),
                "param_name": name,
                "routing": routing,
                "grad_norm": grad_norm,
                "update_norm": update_norm,
                "weight_norm": weight_norm,
                "update_to_weight_ratio": update_norm / (weight_norm + 1e-12),
                "top_singular_value": top_sv,
                "effective_rank": effective_rank,
                "orthogonality_error": orthogonality_error(weight.cpu()),
            }
        )
    return rows


def save_checkpoint(path: Path, model: GPT, adamw, muon, step: int, tokens_seen: int, best_val: float, args) -> None:
    payload = {
        "model": model.state_dict(),
        "adamw": adamw.state_dict() if adamw is not None else None,
        "muon": muon.state_dict() if muon is not None else None,
        "step": step,
        "tokens_seen": tokens_seen,
        "best_val": best_val,
        "args": vars(args),
    }
    torch.save(payload, path)


def load_checkpoint(path: Path, model: GPT, adamw, muon, device: str):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    if adamw is not None and ckpt.get("adamw"):
        adamw.load_state_dict(ckpt["adamw"])
    if muon is not None and ckpt.get("muon"):
        muon.load_state_dict(ckpt["muon"])
    return int(ckpt.get("step", 0)), int(ckpt.get("tokens_seen", 0)), float(ckpt.get("best_val", float("inf")))


@torch.no_grad()
def write_samples(model: GPT, args, run_dir: Path, step: int, checkpoint_path: str | None) -> None:
    enc = get_tokenizer()
    prompts = TINY_PROMPTS if args.dataset == "tinystories" else FINEWEB_PROMPTS
    model.eval()
    gen = torch.Generator(device=args.device)
    gen.manual_seed(12345)
    torch.manual_seed(12345)
    rows = []
    for prompt in prompts:
        ids = torch.tensor([enc.encode_ordinary(prompt)], dtype=torch.long, device=args.device)
        out = model.generate(ids, max_new_tokens=120, temperature=0.8, top_k=50)[0].tolist()
        rows.append(
            {
                "step": step,
                "prompt": prompt,
                "generated_text": enc.decode(out),
                "routing": args.routing,
                "seed": args.seed,
                "checkpoint_path": checkpoint_path,
            }
        )
    with (run_dir / "samples.jsonl").open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    model.train()


def append_csv(path: Path, fieldnames: list[str], row: dict) -> None:
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    if args.prepare_dataset:
        if args.prepare_dataset == "tinystories":
            prepare_tinystories(root)
        else:
            prepare_fineweb(root, args.target_tokens, args.shard_tokens, vars(args))
        return

    seed_everything(args.seed)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but torch.cuda.is_available() is false. "
            "Install a PyTorch wheel compatible with the NVIDIA driver before training."
        )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = root / args.out_dir / f"{timestamp}_{args.dataset}_{args.model_size}_{args.routing}_seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "figures").mkdir()
    (run_dir / "qualitative").mkdir()
    sys.stdout = Tee(sys.stdout, run_dir / "train.log")
    sys.stderr = Tee(sys.stderr, run_dir / "train.log")

    train_data, valid_data = build_datasets(root, args.dataset, args.context_length, args.device)
    cfg_kwargs = MODEL_SIZES[args.model_size] | {"block_size": args.context_length}
    model = GPT(GPTConfig(**cfg_kwargs)).to(args.device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")

    adamw, muon, groups = build_optimizers(model, args, run_dir)
    if args.compile:
        model = torch.compile(model)  # type: ignore[assignment]

    start_step, tokens_seen, best_val = 0, 0, float("inf")
    if args.resume:
        start_step, tokens_seen, best_val = load_checkpoint(Path(args.resume), model, adamw, muon, args.device)
        print(f"Resumed from {args.resume} at step {start_step}, tokens {tokens_seen}")

    resolved = vars(args) | {
        "run_dir": str(run_dir),
        "parameter_count": param_count,
        "git_commit": get_git_commit(),
        "optimizer_group_counts": {"muon": len(groups["muon"]), "adamw": len(groups["adamw"])},
    }
    (run_dir / "config_resolved.yaml").write_text(yaml.safe_dump(resolved, sort_keys=True), encoding="utf-8")
    if args.dry_run:
        print("Dry run complete.")
        return

    amp_dtype = get_amp_dtype(args.dtype, args.device)
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16 and args.device.startswith("cuda")))
    tokens_per_step = args.batch_size * args.context_length * args.grad_accum
    max_steps = max(1, math.ceil(args.max_tokens / tokens_per_step))
    warmup_steps = max(10, int(0.02 * max_steps))
    start_time = time.time()
    metric_fields = [
        "step",
        "tokens_seen",
        "epoch_or_stream_position",
        "train_loss",
        "val_loss",
        "lr_adamw",
        "lr_muon",
        "grad_norm_global",
        "update_norm_global",
        "tokens_per_sec",
        "wall_clock_seconds",
        "gpu_memory_allocated_gb",
        "gpu_memory_reserved_gb",
    ]
    geom_fields = [
        "step",
        "tokens_seen",
        "module_name",
        "param_name",
        "routing",
        "grad_norm",
        "update_norm",
        "weight_norm",
        "update_to_weight_ratio",
        "top_singular_value",
        "effective_rank",
        "orthogonality_error",
    ]

    model.train()
    for step in range(start_step + 1, max_steps + 1):
        lr_adamw = cosine_lr(args.adamw_lr, step, max_steps, warmup_steps)
        lr_muon = cosine_lr(args.muon_lr, step, max_steps, warmup_steps)
        set_lr(adamw, lr_adamw)
        set_lr(muon, lr_muon)
        adamw.zero_grad(set_to_none=True)
        if muon is not None:
            muon.zero_grad(set_to_none=True)
        last_loss = 0.0
        for _ in range(args.grad_accum):
            x, y = train_data.get_batch(args.batch_size)
            with torch.autocast(device_type=args.device.split(":")[0], dtype=amp_dtype, enabled=amp_dtype != torch.float32):
                _, loss = model(x, y)
                loss = loss / args.grad_accum
            last_loss = float(loss.item() * args.grad_accum)
            scaler.scale(loss).backward()

        scaler.unscale_(adamw)
        if muon is not None:
            scaler.unscale_(muon)
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item())
        scaler.step(adamw)
        if muon is not None:
            scaler.step(muon)
        scaler.update()

        tokens_seen += tokens_per_step
        elapsed = time.time() - start_time
        do_eval = step == 1 or step % args.eval_interval == 0 or step == max_steps
        val_loss = float("nan")
        train_eval_loss = last_loss
        if do_eval:
            train_eval_loss, val_loss = estimate_loss(model, train_data, valid_data, args, amp_dtype)
            ckpt_last = run_dir / "checkpoint_last.pt"
            save_checkpoint(ckpt_last, model, adamw, muon, step, tokens_seen, best_val, args)
            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(run_dir / "checkpoint_best.pt", model, adamw, muon, step, tokens_seen, best_val, args)

        update_norm = 0.0
        for p in model.parameters():
            update = getattr(p, "_muon_last_update", None)
            if update is not None:
                update_norm += float(update.norm().item() ** 2)
        update_norm = update_norm**0.5
        mem_alloc = torch.cuda.memory_allocated() / 1e9 if args.device.startswith("cuda") else 0.0
        mem_reserved = torch.cuda.memory_reserved() / 1e9 if args.device.startswith("cuda") else 0.0
        append_csv(
            run_dir / "metrics.csv",
            metric_fields,
            {
                "step": step,
                "tokens_seen": tokens_seen,
                "epoch_or_stream_position": tokens_seen,
                "train_loss": train_eval_loss,
                "val_loss": val_loss,
                "lr_adamw": lr_adamw,
                "lr_muon": lr_muon if muon is not None else 0.0,
                "grad_norm_global": grad_norm,
                "update_norm_global": update_norm,
                "tokens_per_sec": tokens_seen / max(elapsed, 1e-9),
                "wall_clock_seconds": elapsed,
                "gpu_memory_allocated_gb": mem_alloc,
                "gpu_memory_reserved_gb": mem_reserved,
            },
        )

        if step % args.geometry_interval == 0 or step == max_steps:
            for row in collect_geometry(model, step, tokens_seen, args.routing):
                append_csv(run_dir / "geometry_metrics.csv", geom_fields, row)
        if step % args.sample_interval == 0 or step == max_steps:
            write_samples(model, args, run_dir, step, str(run_dir / "checkpoint_last.pt"))
        if step % args.save_every == 0:
            save_checkpoint(run_dir / "checkpoint_last.pt", model, adamw, muon, step, tokens_seen, best_val, args)

        val_display = f"{val_loss:.4f}" if not math.isnan(val_loss) else "-"
        print(
            f"step {step}/{max_steps} tokens {tokens_seen:,} train {last_loss:.4f} "
            f"val {val_display} tok/s {tokens_seen / max(elapsed, 1e-9):.0f}"
        )
        if tokens_seen >= args.max_tokens:
            break

    save_checkpoint(run_dir / "checkpoint_last.pt", model, adamw, muon, step, tokens_seen, best_val, args)
    write_samples(model, args, run_dir, step, str(run_dir / "checkpoint_last.pt"))
    print(f"Finished run in {run_dir}")


class Tee:
    def __init__(self, stream, path: Path) -> None:
        self.stream = stream
        self.file = path.open("a", encoding="utf-8")

    def write(self, data: str) -> int:
        self.stream.write(data)
        self.stream.flush()
        self.file.write(data)
        self.file.flush()
        return len(data)

    def flush(self) -> None:
        self.stream.flush()
        self.file.flush()


if __name__ == "__main__":
    main()
