"""The full reasoning-loop wrapper: N_in -> R -> (S -> R)*T -> N_out."""
from __future__ import annotations
import torch
import torch.nn as nn

from .config import BoundaryConfig
from .boundary import _call_layer
from .standardization import Standardization


class ReasoningLoopModel(nn.Module):
    """A reasoning-body / standardisation-module wrapper around a base causal LM.

    The base model is fully frozen. The only trainable parameters live in the
    standardizations' LoRA adapters and their core_S modules.

    `k_in` and `k_out` are owned by the model; every standardization shares them.
    Each standardization independently chooses, within that boundary region,
    which inner slots get LoRA (via its `BoundaryConfig.ranks` list). Outer
    slots beyond the rank list length are skipped implicitly.

    Stage decomposition for a base model with L transformer layers:
        encode    : embed + layers[0 : L - k_out]      (clean base weights)
        loop_step : standardizations[idx] + layers[k_in : L - k_out]
                    (the active standardization's boundaries apply LoRA)
        decode    : layers[L - k_out : L] + norm + lm_head  (clean base weights)

    Integration notes:
      - **Layer kwargs**: `**kw` on forward/encode/loop_step/decode is threaded
        through to every base-layer call via `_call_layer`. Pass attention_mask,
        position_ids, cache_position, etc. via **kw. The framework does not
        own or update those values.
      - **KV cache**: do NOT pass `past_key_values` or `use_cache=True` to a
        forward() call that performs loop iterations (schedule non-empty). Each
        loop step re-processes the same sequence positions, which would
        corrupt the cache. For autoregressive generation, manage the cache
        externally and call forward repeatedly on growing inputs.
      - **FSDP**: `_base_layers` is a plain Python list inside each Boundary,
        so base-model parameters are owned exclusively by `self.base`.
        `model.parameters()` yields base params + per-standardization LoRA
        deltas + per-standardization core_S params, each exactly once. The
        fully_shard plan should target `self.base.model.layers[i]` and each
        `self.standardizations[j]` independently.
      - **torch.compile**: `Boundary.forward` is marked with
        `torch.compiler.disable` (the hook-based LoRA activation cannot be
        traced). encode, decode, and the body-iteration portion of loop_step
        compile cleanly; each Standardization call inside loop_step becomes
        a graph break.
    """

    #: Valid arguments to `set_freeze_mode`.
    FREEZE_MODES = ("reasoning", "standardization", "none")

    def __init__(self, base_model: nn.Module, k_in: int, k_out: int,
                 standardizations: list[tuple[BoundaryConfig, BoundaryConfig, nn.Module]]):
        super().__init__()
        L = len(base_model.model.layers)
        if k_in + k_out > L:
            raise ValueError(
                f"k_in + k_out = {k_in + k_out} exceeds base depth L = {L}"
            )
        if not standardizations:
            raise ValueError("standardizations must be non-empty")
        for i, (head_cfg, tail_cfg, _) in enumerate(standardizations):
            if len(head_cfg.ranks) > k_in:
                raise ValueError(
                    f"standardizations[{i}]: len(head_cfg.ranks)={len(head_cfg.ranks)} "
                    f"exceeds k_in={k_in}"
                )
            if len(tail_cfg.ranks) > k_out:
                raise ValueError(
                    f"standardizations[{i}]: len(tail_cfg.ranks)={len(tail_cfg.ranks)} "
                    f"exceeds k_out={k_out}"
                )

        self.base = base_model
        self.k_in = k_in
        self.k_out = k_out
        for p in self.base.parameters():
            p.requires_grad_(False)

        self.standardizations = nn.ModuleList([
            Standardization(base_model, head_cfg, tail_cfg, core_S, k_in, k_out)
            for head_cfg, tail_cfg, core_S in standardizations
        ])

        # Default state: frozen body, trainable standardization — the same
        # partition the base freeze above already produces. Recorded so the
        # freeze mode is always introspectable and set explicitly.
        self.freeze_mode = "reasoning"

    # ------------------------------------------------------------------
    # Stage primitives
    # ------------------------------------------------------------------

    @property
    def L(self) -> int:
        return len(self.base.model.layers)

    def encode(self, input_ids: torch.Tensor, **kw) -> torch.Tensor:
        """N_in then R: embed + layers[0 : L - k_out]."""
        h = self.base.model.embed_tokens(input_ids)
        for layer in self.base.model.layers[: self.L - self.k_out]:
            h = _call_layer(layer, h, **kw)
        return h

    def loop_step(self, h: torch.Tensor, idx: int, **kw) -> torch.Tensor:
        """One iteration: standardizations[idx] then R."""
        h = self.standardizations[idx](h, **kw)
        for layer in self.base.model.layers[self.k_in : self.L - self.k_out]:
            h = _call_layer(layer, h, **kw)
        return h

    def decode(self, h: torch.Tensor, **kw) -> torch.Tensor:
        """N_out: layers[L - k_out : L] + final norm (if present) + lm_head."""
        for layer in self.base.model.layers[self.L - self.k_out :]:
            h = _call_layer(layer, h, **kw)
        if hasattr(self.base.model, "norm"):
            h = self.base.model.norm(h)
        return self.base.lm_head(h)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def forward(self, input_ids: torch.Tensor, schedule: list[int],
                **kw) -> torch.Tensor:
        """Full pipeline: encode -> loop_step for each idx in schedule -> decode."""
        n = len(self.standardizations)
        for i, idx in enumerate(schedule):
            if not (0 <= idx < n):
                raise IndexError(
                    f"schedule[{i}]={idx} out of range [0,{n})"
                )
        h = self.encode(input_ids, **kw)
        for idx in schedule:
            h = self.loop_step(h, idx, **kw)
        return self.decode(h, **kw)

    def trainable_parameters(self):
        return (p for p in self.parameters() if p.requires_grad)

    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.trainable_parameters())

    def total_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    # ------------------------------------------------------------------
    # Freeze mode
    # ------------------------------------------------------------------

    def reasoning_parameters(self):
        """Parameters of the reasoning body: base layers[k_in : L - k_out]."""
        for layer in self.base.model.layers[self.k_in : self.L - self.k_out]:
            yield from layer.parameters()

    def standardization_parameters(self):
        """Standardization-owned parameters: LoRA adapters + each core_S's own
        params, excluding any base parameter a core_S merely references (e.g.
        embed / norm / lm_head, or a reused body layer). Those base-shared
        params stay governed by the frozen base, never by this component."""
        base_ids = {id(p) for p in self.base.parameters()}
        for p in self.standardizations.parameters():
            if id(p) not in base_ids:
                yield p

    def set_freeze_mode(self, mode: str) -> None:
        """Toggle which trainable-eligible component is frozen.

        Governs exactly two parameter groups:
          - reasoning:      base layers[k_in : L - k_out]   (see reasoning_parameters)
          - standardization: LoRA adapters + core_S own params (see
                             standardization_parameters)

        Modes:
          - "reasoning":       freeze the reasoning body, train the standardization
          - "standardization": freeze the standardization, train the reasoning body
          - "none":            train both

        The base's non-body parameters — the first k_in and last k_out boundary
        layers, plus embed_tokens / norm / lm_head, and any base param a core_S
        references — are NOT governed here and remain frozen (as set at
        construction, or as `BoundaryConfig.train_base` left them). Consequently
        "none" only unfreezes *all* transformer layers when k_in = k_out = 0;
        otherwise the boundary layers stay frozen.
        """
        if mode not in self.FREEZE_MODES:
            raise ValueError(
                f"mode must be one of {self.FREEZE_MODES}, got {mode!r}"
            )
        train_reasoning = mode in ("standardization", "none")
        train_standardization = mode in ("reasoning", "none")
        for p in self.reasoning_parameters():
            p.requires_grad_(train_reasoning)
        for p in self.standardization_parameters():
            p.requires_grad_(train_standardization)
        self.freeze_mode = mode
