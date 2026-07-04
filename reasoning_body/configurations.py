"""Canonical configurations from the method-mapping table.

Each constructor returns (head_cfg, tail_cfg, core_factory) where
core_factory(base_model) -> nn.Module produces the appropriate core_S given
access to the base model's embed_tokens and lm_head.
"""
from __future__ import annotations
from typing import Callable

import torch.nn as nn

from . import config as cfg
from .core import Identity, ExactNextToken, SoftmaxEmbedMixture, MLPCore, Codebook


CoreFactory = Callable[[nn.Module], nn.Module]
Configuration = tuple[cfg.BoundaryConfig, cfg.BoundaryConfig, CoreFactory]


# ---------------------------------------------------------------------------
# Existing methods, mapped into the framework
# ---------------------------------------------------------------------------

def coconut() -> Configuration:
    """COCONUT: k_in = k_out = 0, S = identity."""
    return cfg.empty(), cfg.empty(), lambda _m: Identity()


def cot() -> Configuration:
    """CoT: k_in = k_out = 0, S = exact greedy next-token re-embed."""
    return (cfg.empty(), cfg.empty(),
            lambda m: ExactNextToken(getattr(m.model, "norm", None),
                                     m.lm_head, m.model.embed_tokens))


def soft_thinking(temperature: float = 1.0) -> Configuration:
    """Soft Thinking: k_in = k_out = 0, S = vocabulary-weighted mixture."""
    return (cfg.empty(), cfg.empty(),
            lambda m: SoftmaxEmbedMixture(m.lm_head, m.model.embed_tokens, temperature))


# ---------------------------------------------------------------------------
# Novel configurations enabled by the framework
# ---------------------------------------------------------------------------

def frozen_baseline(k_in: int = 4, k_out: int = 4,
                    train_base: bool = False) -> Configuration:
    """k boundary layers, no LoRA. With train_base=True the base layer
    weights are trainable — the natural setup for CoT-style fine-tuning of
    the boundary region with the body kept frozen."""
    return (cfg.frozen(k_in, train_base=train_base),
            cfg.frozen(k_out, train_base=train_base),
            lambda _m: Identity())


def uniform_lora(k_in: int = 4, k_out: int = 4, rank: int = 8,
                 init: str = "zero",
                 train_base: bool = False) -> Configuration:
    """Same LoRA rank on every in-loop boundary layer."""
    return (cfg.uniform(k_in, rank=rank, init=init, train_base=train_base),
            cfg.uniform(k_out, rank=rank, init=init, train_base=train_base),
            lambda _m: Identity())


def inner_heavy_lora(k_in: int = 4, k_out: int = 4,
                     max_rank: int = 16, decay: int = 2,
                     init: str = "zero",
                     train_base: bool = False) -> Configuration:
    """Peak LoRA rank at body-adjacent slot, decreasing outward. Identity core."""
    return (cfg.inner_heavy(k_in, max_rank=max_rank, decay=decay,
                            init=init, train_base=train_base),
            cfg.inner_heavy(k_out, max_rank=max_rank, decay=decay,
                            init=init, train_base=train_base),
            lambda _m: Identity())


def outer_heavy_lora(k_in: int = 4, k_out: int = 4,
                     max_rank: int = 16, decay: int = 2,
                     init: str = "zero",
                     train_base: bool = False) -> Configuration:
    """Peak LoRA rank at vocab-adjacent slot, decreasing inward. Identity core."""
    return (cfg.outer_heavy(k_in, max_rank=max_rank, decay=decay,
                            init=init, train_base=train_base),
            cfg.outer_heavy(k_out, max_rank=max_rank, decay=decay,
                            init=init, train_base=train_base),
            lambda _m: Identity())


def skip_outer_lora(k_in: int = 4, k_out: int = 4,
                    num_inner: int = 2, rank: int = 8,
                    init: str = "zero",
                    train_base: bool = False) -> Configuration:
    """Adapt only the slots closest to the body; skip the rest."""
    return (cfg.skip_outer(k_in, num_inner=num_inner, rank=rank,
                           init=init, train_base=train_base),
            cfg.skip_outer(k_out, num_inner=num_inner, rank=rank,
                           init=init, train_base=train_base),
            lambda _m: Identity())


def inner_heavy_soft(k_in: int = 4, k_out: int = 4,
                     max_rank: int = 8, decay: int = 2,
                     temperature: float = 1.0,
                     init: str = "zero",
                     train_base: bool = False) -> Configuration:
    """Inner-heavy LoRA boundary + Soft Thinking core. Tests complementarity."""
    return (cfg.inner_heavy(k_in, max_rank=max_rank, decay=decay,
                            init=init, train_base=train_base),
            cfg.inner_heavy(k_out, max_rank=max_rank, decay=decay,
                            init=init, train_base=train_base),
            lambda m: SoftmaxEmbedMixture(m.lm_head, m.model.embed_tokens, temperature))


def mlp_core(k_in: int = 0, k_out: int = 0,
             expansion: float = 2.0) -> Configuration:
    """Learned MLP core, no boundary layers. Pure learned standardisation."""
    return (cfg.empty() if k_in == 0 else cfg.frozen(k_in),
            cfg.empty() if k_out == 0 else cfg.frozen(k_out),
            lambda m: MLPCore(m.model.embed_tokens.embedding_dim, expansion=expansion))


def codebook_core(k_in: int = 0, k_out: int = 0,
                  num_codes: int = 512) -> Configuration:
    """Discrete codebook core, no boundary layers."""
    return (cfg.empty() if k_in == 0 else cfg.frozen(k_in),
            cfg.empty() if k_out == 0 else cfg.frozen(k_out),
            lambda m: Codebook(m.model.embed_tokens.embedding_dim, num_codes=num_codes))
