"""The Boundary module: per-layer routing of LoRA-adapted / frozen slots.

The Boundary holds non-owning references to base-model layers in a plain Python
list (`_base_layers`) so that PyTorch parameter discovery and FSDP sharding plans
do not double-count them. The Boundary's only owned parameters are the LoRA
deltas it constructs for slots with rank > 0.

A `BoundaryConfig.ranks` of length n < k means the outer (k - n) slots are
skipped implicitly. Skipping happens at `build_boundary` time by selecting only
the innermost n base layers; the Boundary itself only sees the n slots it
operates on.
"""
from __future__ import annotations
from typing import Literal

import torch
import torch.nn as nn

from .config import BoundaryConfig
from .lora import build_lora_pack, lora_active

# The hook-based LoRA activation in `lora_active` mutates module state mid-call
# (register/remove forward hooks), which is hostile to TorchDynamo tracing.
# We mark `Boundary.forward` as a hard graph break: dynamo runs it eagerly and
# resumes tracing on either side. This keeps the body iteration in
# `ReasoningLoopModel.loop_step` (outside the Boundary) inside the compiled graph.
try:
    from torch.compiler import disable as _compile_disable
except ImportError:  # torch < 2.1
    def _compile_disable(fn):  # type: ignore[misc]
        return fn


def _call_layer(layer: nn.Module, h: torch.Tensor, **kwargs) -> torch.Tensor:
    """Invoke a (possibly HF-style) decoder layer that may return a tuple."""
    out = layer(h, **kwargs)
    return out[0] if isinstance(out, tuple) else out


class Boundary(nn.Module):
    """Trainable interface between core_S and the reasoning body.

    Indexing convention (same as BoundaryConfig.ranks):
        ranks[0]    = innermost (adjacent to body)
        ranks[-1]   = outermost slot in this Boundary
    `_base_layers[i]` corresponds to `cfg.ranks[i]`.

    Execution order for a single forward pass:
        side="tail": innermost (i=0) -> outermost (i=n-1)
        side="head": outermost (i=n-1) -> innermost (i=0)
    matching the data flow through the physical layer sequence of the base model.
    """

    def __init__(self, base_layers: list[nn.Module], cfg: BoundaryConfig,
                 side: Literal["head", "tail"]):
        super().__init__()
        if len(base_layers) != len(cfg.ranks):
            raise ValueError(
                f"len(base_layers)={len(base_layers)} != len(cfg.ranks)={len(cfg.ranks)}"
            )
        if side not in ("head", "tail"):
            raise ValueError(f"side must be 'head' or 'tail', got {side!r}")

        # NOT an nn.ModuleList: we intentionally do not register base layers
        # as submodules so they remain owned by the base model only.
        self._base_layers: list[nn.Module] = list(base_layers)
        self.side = side
        self.cfg = cfg

        # adapters[i] is a ModuleDict (possibly empty) for slot i.
        self.adapters = nn.ModuleList([
            build_lora_pack(layer, rank, cfg.alpha, cfg.targets, init=cfg.init)
            if rank > 0 else nn.ModuleDict()
            for layer, rank in zip(base_layers, cfg.ranks)
        ])

        # Optionally unfreeze the base-layer parameters for end-to-end training
        # (e.g. CoT-style fine-tuning of the boundary layers, in addition to
        # any LoRA delta). The ReasoningLoopModel freezes the full base model
        # at construction time *before* Boundaries are built, so this flips
        # the relevant parameters back on. Body layers are never in any
        # Boundary's _base_layers, so they stay frozen.
        if cfg.train_base:
            for layer in self._base_layers:
                for p in layer.parameters():
                    p.requires_grad_(True)

    @_compile_disable
    def forward(self, hidden_states: torch.Tensor, **layer_kwargs) -> torch.Tensor:
        n = len(self.cfg.ranks)
        order = range(n) if self.side == "tail" else range(n - 1, -1, -1)
        for i in order:
            rank = self.cfg.ranks[i]
            layer = self._base_layers[i]
            if rank > 0:
                with lora_active(layer, self.adapters[i], self.cfg.targets):
                    hidden_states = _call_layer(layer, hidden_states, **layer_kwargs)
            else:
                hidden_states = _call_layer(layer, hidden_states, **layer_kwargs)
        return hidden_states


def build_boundary(cfg: BoundaryConfig, base_model: nn.Module,
                   side: Literal["head", "tail"], k: int) -> Boundary | None:
    """Construct a Boundary by selecting the innermost len(cfg.ranks) of the
    k boundary layers on the given side.

    Returns None if cfg.ranks is empty (the boundary side is absent).
    Raises ValueError if len(cfg.ranks) > k.
    """
    n = len(cfg.ranks)
    if n == 0:
        return None
    if n > k:
        raise ValueError(f"len(cfg.ranks)={n} exceeds boundary size k={k}")
    base_layers_all = base_model.model.layers
    L = len(base_layers_all)
    if side == "tail":
        # The tail region occupies layers [L - k, L). Its innermost slot
        # (adjacent to the body) is layer L - k. We use only the innermost n.
        base_layers = [base_layers_all[L - k + i] for i in range(n)]
    else:  # head
        # The head region occupies layers [0, k). Its innermost slot
        # (adjacent to the body) is layer k - 1. We use only the innermost n.
        base_layers = [base_layers_all[k - 1 - i] for i in range(n)]
    return Boundary(base_layers, cfg, side)
