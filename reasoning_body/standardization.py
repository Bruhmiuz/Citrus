"""Standardization S = tail -> core_S -> head, registered as an nn.Module.

Each Standardization owns its own head/tail Boundaries (with independent LoRA
adapters) and its own core_S. Multiple Standardizations may coexist over the
same base layers within a single ReasoningLoopModel; LoRA hooks are installed
only during the active Boundary's forward, so adapters do not alias.
"""
from __future__ import annotations
import torch
import torch.nn as nn

from .config import BoundaryConfig
from .boundary import build_boundary


class Standardization(nn.Module):
    def __init__(self, base_model: nn.Module,
                 head_cfg: BoundaryConfig,
                 tail_cfg: BoundaryConfig,
                 core_S: nn.Module,
                 k_in: int, k_out: int):
        super().__init__()
        self.head_cfg = head_cfg
        self.tail_cfg = tail_cfg
        self.tail = build_boundary(tail_cfg, base_model, "tail", k_out)
        self.core_S = core_S
        self.head = build_boundary(head_cfg, base_model, "head", k_in)

    def forward(self, h: torch.Tensor, **kw) -> torch.Tensor:
        if self.tail is not None:
            h = self.tail(h, **kw)
        h = self.core_S(h)
        if self.head is not None:
            h = self.head(h, **kw)
        return h
