"""Minimal Muon optimizer used by Muon Routing Atlas.

This is a compact local implementation inspired by Keller Jordan's reference
Muon code and writeup:
https://github.com/KellerJordan/Muon
https://kellerjordan.github.io/posts/muon/

It is intentionally limited to single-process, single-GPU research runs. Muon
is only meant to receive selected hidden 2D matrices; embeddings, heads, norms,
and biases should stay in AdamW in ``train_muon_atlas.py``.
"""

from __future__ import annotations

from typing import Iterable

import torch
from torch.optim import Optimizer


@torch.no_grad()
def zeropower_via_newtonschulz5(matrix: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Approximate the polar factor of a 2D update matrix.

    The coefficients follow the widely used Muon Newton-Schulz iteration. The
    output is approximately orthogonalized along the smaller matrix dimension.
    """

    if matrix.ndim != 2:
        raise ValueError(f"Muon orthogonalization expects a 2D tensor, got shape {tuple(matrix.shape)}")

    x = matrix.float()
    if x.numel() == 0:
        return x.to(dtype=matrix.dtype)
    x = x / (x.norm() + eps)
    transposed = False
    if x.shape[0] > x.shape[1]:
        x = x.T
        transposed = True

    # Quintic Newton-Schulz coefficients used in the reference Muon optimizer.
    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(steps):
        xx_t = x @ x.T
        x = a * x + (b * xx_t + c * xx_t @ xx_t) @ x

    if transposed:
        x = x.T
    return x.to(dtype=matrix.dtype)


class Muon(Optimizer):
    """Orthogonalized momentum optimizer for selected 2D matrices."""

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 0.02,
        momentum: float = 0.95,
        weight_decay: float = 0.05,
        ns_steps: int = 5,
    ) -> None:
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            weight_decay = group["weight_decay"]
            ns_steps = group["ns_steps"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.ndim != 2:
                    raise ValueError(f"Muon received non-2D parameter with shape {tuple(p.shape)}")

                grad = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(grad)

                update = zeropower_via_newtonschulz5(buf, steps=ns_steps)
                scale = max(1.0, p.shape[0] / max(1, p.shape[1])) ** 0.5
                if weight_decay:
                    p.mul_(1.0 - lr * weight_decay)
                delta = -lr * scale * update
                p.add_(delta)
                p._muon_last_update = delta.detach().clone()  # used for sparse geometry logging

        return loss
