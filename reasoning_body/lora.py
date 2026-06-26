"""Minimal LoRA implementation with hook-based activation.

LoRA deltas are owned by the Boundary (not the base layer). They are applied
only when the Boundary's forward pass runs, via temporary forward hooks on the
targeted submodules. This keeps the base layers' parameters clean during the
encode/decode passes that happen outside the loop.
"""
from __future__ import annotations
import math
from contextlib import contextmanager
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


# ModuleDict keys can't contain dots, so we mangle "self_attn.q_proj" -> "self_attn__q_proj".
def _mangle(path: str) -> str:
    return path.replace(".", "__")


class LoRADelta(nn.Module):
    """Rank-r additive update: y += (alpha / r) * B @ A @ x.

    Standard initialisation: A ~ Kaiming, B = 0, so the initial delta is exactly
    zero and the wrapped layer behaves identically to the frozen base layer.
    """
    def __init__(self, in_features: int, out_features: int,
                 rank: int, alpha: float):
        super().__init__()
        if rank <= 0:
            raise ValueError(f"rank must be positive, got {rank}")
        self.A = nn.Parameter(torch.empty(rank, in_features))
        self.B = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.scale = alpha / rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(F.linear(x, self.A), self.B) * self.scale


def _resolve_module(root: nn.Module, path: str) -> nn.Module:
    obj: nn.Module = root
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj


def _init_soft_skip(delta: "LoRADelta", target_weight: torch.Tensor) -> None:
    """Re-initialise A, B so that scale * B @ A approximately equals -W_topr,
    the best rank-r approximation of the negated base weight.

    Effect at init: the wrapped projection (W + scale * B @ A) x is approximately
    (W - W_topr) x, leaving only the residual small-singular-value components
    of the base layer. Training can move BA away from this anti-init.

    Only the linear weight is cancelled; bias (if any) is untouched.
    """
    r = delta.A.shape[0]
    # torch.linalg.svd requires float32+; base weight may be bf16/fp16.
    W = target_weight.detach().to(torch.float32)
    U, S, Vh = torch.linalg.svd(-W, full_matrices=False)
    if r > S.shape[0]:
        raise ValueError(
            f"soft_skip: rank {r} exceeds min(out, in) = {S.shape[0]} of target weight"
        )
    U_r = U[:, :r]                                    # [out, r]
    S_r = S[:r]                                       # [r]
    Vh_r = Vh[:r, :]                                  # [r, in]
    d = torch.sqrt(torch.clamp(S_r, min=0.0) / delta.scale)
    A_init = d[:, None] * Vh_r                        # [r, in]
    B_init = U_r * d[None, :]                         # [out, r]
    with torch.no_grad():
        delta.A.copy_(A_init.to(delta.A.dtype))
        delta.B.copy_(B_init.to(delta.B.dtype))


def build_lora_pack(layer: nn.Module, rank: int, alpha: float,
                    targets: Iterable[str], init: str = "zero") -> nn.ModuleDict:
    """Construct a ModuleDict of LoRADeltas, one per targeted submodule of `layer`.

    init="zero":      A ~ Kaiming, B = 0 (initial delta = 0, base layer unchanged).
    init="soft_skip": A, B initialised so scale * B @ A approximately equals -W_topr
                      (partial cancellation of the targeted projection at init).
    """
    pack = nn.ModuleDict()
    for path in targets:
        target = _resolve_module(layer, path)
        if not isinstance(target, nn.Linear):
            raise TypeError(
                f"LoRA target '{path}' must be nn.Linear, got {type(target).__name__}"
            )
        delta = LoRADelta(target.in_features, target.out_features, rank, alpha)
        if init == "soft_skip":
            _init_soft_skip(delta, target.weight)
        elif init != "zero":
            raise ValueError(f"unknown init mode: {init!r}")
        pack[_mangle(path)] = delta
    return pack


def _make_lora_hook(delta: LoRADelta):
    def hook(_module: nn.Module, inputs: tuple, output: torch.Tensor) -> torch.Tensor:
        return output + delta(inputs[0])
    return hook


@contextmanager
def lora_active(layer: nn.Module, pack: nn.ModuleDict, target_paths: Iterable[str]):
    """Temporarily install LoRA forward hooks on the targeted submodules of `layer`.

    On exit (including exceptions), hooks are removed and the layer is restored.
    """
    handles = []
    try:
        for path in target_paths:
            target = _resolve_module(layer, path)
            delta = pack[_mangle(path)]
            handles.append(target.register_forward_hook(_make_lora_hook(delta)))
        yield
    finally:
        for h in handles:
            h.remove()
