"""Configuration for the standardisation module.

A boundary side (head or tail) is described by a `BoundaryConfig` carrying a
flat list of LoRA ranks indexed innermost (adjacent to the body) -> outermost
(adjacent to embedding / unembedding). For a model with boundary size `k`:

    ranks[i] > 0   -> trainable LoRA adapter of that rank on the i-th slot
    ranks[i] == 0  -> base layer runs in the loop, frozen, no delta
    i >= len(ranks)-> slot is implicitly skipped (layer bypassed in the loop)

The boundary size `k` is owned by `ReasoningLoopModel`, not by the config.
A config's rank list may be shorter than the model's `k_in`/`k_out`; the
outer slots are then skipped.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal


# Default LoRA targets for Llama-style decoder layers.
DEFAULT_LORA_TARGETS: tuple[str, ...] = (
    "self_attn.q_proj",
    "self_attn.v_proj",
    "mlp.down_proj",
)

InitMode = Literal["zero", "soft_skip"]


@dataclass
class BoundaryConfig:
    ranks: list[int] = field(default_factory=list)
    alpha: float = 16.0
    targets: tuple[str, ...] = DEFAULT_LORA_TARGETS
    init: InitMode = "zero"
    train_base: bool = False
    """If True, the Boundary unfreezes the parameters of every base layer it
    holds — both rank=0 slots (frozen-but-now-trainable) and rank>0 slots
    (trainable LoRA stacked on a trainable base). Lets the model replicate
    standard fine-tuning (CoT-style) within the boundary region while keeping
    the body frozen. Note: base layers are shared across Standardizations, so
    if ANY Boundary over a given layer sets train_base=True, that layer is
    trainable for every Standardization."""

    def __post_init__(self) -> None:
        if any(r < 0 for r in self.ranks):
            raise ValueError("ranks must be >= 0")
        if self.init not in ("zero", "soft_skip"):
            raise ValueError(f"init must be 'zero' or 'soft_skip', got {self.init!r}")


# ---------------------------------------------------------------------------
# Gradient-shape constructors. `k` controls the length of the produced rank
# list (i.e., how many inner slots the config activates). The model's
# `k_in`/`k_out` must be >= `k`; outer slots beyond `k` are skipped implicitly.
# ---------------------------------------------------------------------------

def empty() -> BoundaryConfig:
    """Empty ranks list. The entire boundary side is absent."""
    return BoundaryConfig()


def frozen(k: int, targets: tuple[str, ...] = DEFAULT_LORA_TARGETS,
           train_base: bool = False) -> BoundaryConfig:
    """k inner slots, no LoRA. By default no trainable params; with
    train_base=True the base layer weights themselves become trainable."""
    return BoundaryConfig(ranks=[0] * k, targets=targets, train_base=train_base)


def uniform(k: int, rank: int = 8, alpha: float = 16.0,
            targets: tuple[str, ...] = DEFAULT_LORA_TARGETS,
            init: InitMode = "zero",
            train_base: bool = False) -> BoundaryConfig:
    """Same LoRA rank on every in-loop slot."""
    return BoundaryConfig(ranks=[rank] * k, alpha=alpha, targets=targets,
                          init=init, train_base=train_base)


def inner_heavy(k: int, max_rank: int = 16, decay: int = 2, alpha: float = 16.0,
                targets: tuple[str, ...] = DEFAULT_LORA_TARGETS,
                init: InitMode = "zero",
                train_base: bool = False) -> BoundaryConfig:
    """Peak LoRA rank at the body-adjacent slot, decreasing outward.

    Slot i (distance i from the body) gets rank = max_rank // decay**i.
    Predicted to suit vocab-grounded cores (Soft Thinking, ArgmaxResample)
    where core_S output already fits the outer boundary and inner layers
    do the work of re-fitting to body format.
    """
    return BoundaryConfig(
        ranks=[max(max_rank // (decay ** i), 0) for i in range(k)],
        alpha=alpha,
        targets=targets,
        init=init,
        train_base=train_base,
    )


def outer_heavy(k: int, max_rank: int = 16, decay: int = 2, alpha: float = 16.0,
                targets: tuple[str, ...] = DEFAULT_LORA_TARGETS,
                init: InitMode = "zero",
                train_base: bool = False) -> BoundaryConfig:
    """Peak LoRA rank at the vocab-adjacent slot, decreasing inward.

    Predicted to suit Identity / body-format cores, where the outer
    boundary layers receive an input format they were not trained on and
    need the most capacity to compensate.
    """
    return BoundaryConfig(
        ranks=[max(max_rank // (decay ** (k - 1 - i)), 0) for i in range(k)],
        alpha=alpha,
        targets=targets,
        init=init,
        train_base=train_base,
    )


def skip_outer(k: int, num_inner: int, rank: int = 8, alpha: float = 16.0,
               targets: tuple[str, ...] = DEFAULT_LORA_TARGETS,
               init: InitMode = "zero",
               train_base: bool = False) -> BoundaryConfig:
    """Adapt the `num_inner` slots closest to the body; the outer `k - num_inner`
    slots are skipped implicitly by the shortened rank list."""
    if num_inner > k:
        raise ValueError("num_inner must be <= k")
    return BoundaryConfig(ranks=[rank] * num_inner, alpha=alpha, targets=targets,
                          init=init, train_base=train_base)
