"""Implementations of the core standardisation function S-hat.

Each Core maps a hidden state (B, S, D) -> (B, S, D). The choice of Core,
combined with k_in / k_out and the per-slot LoRA configuration, parameterises
the full design space of standardisation modules.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class Identity(nn.Module):
    """COCONUT-style core: pass the hidden state through unchanged.

    Combined with k_in = k_out = 0, recovers Coconut exactly.
    """
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return h


class ExactNextToken(nn.Module):
    """CoT-style core: re-embed the model's exact greedy next-token prediction.

    Projects the hidden state to the vocabulary through the base model's
    genuine prediction head — the final norm followed by the unembedding,
    exactly as ReasoningLoopModel.decode does — then argmaxes and re-embeds:

        S(h) = embed( argmax( lm_head( norm(h) ) ) )

    Running `norm` before `lm_head` is what makes the chosen token the model's
    true greedy prediction rather than an un-normed approximation. `norm` may
    be None for base models with no final normalisation layer.

    Non-differentiable through the argmax; used for inference-time
    instantiations or as a CoT baseline. Per-position: it does not shift,
    grow, or preserve prompt tokens — autoregressive-CoT sequence semantics
    live in the loop, not the core. "Exact" refers to the prediction *head*;
    whether the hidden state reaching this core is the model's true
    final-layer state depends on the tail Boundary spanning the full tail
    region (or k_out = 0).
    """
    def __init__(self, norm: nn.Module | None, unembed: nn.Linear,
                 embed: nn.Embedding):
        super().__init__()
        self.norm = norm
        self.unembed = unembed
        self.embed = embed

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        if self.norm is not None:
            h = self.norm(h)
        logits = self.unembed(h)
        ids = logits.argmax(dim=-1)
        return self.embed(ids)


class SoftmaxEmbedMixture(nn.Module):
    """Soft Thinking core: probability-weighted mixture of token embeddings.

        S(h) = softmax(W_unembed @ h / tau) @ W_embed
    """
    def __init__(self, unembed: nn.Linear, embed: nn.Embedding,
                 temperature: float = 1.0):
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.unembed = unembed
        self.embed = embed
        self.temperature = temperature

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        logits = self.unembed(h) / self.temperature
        probs = F.softmax(logits, dim=-1)
        return probs @ self.embed.weight


class MLPCore(nn.Module):
    """Learned hidden-state-to-hidden-state map with no vocabulary grounding.

    Initialised as identity (the output linear is zero-init), so training
    starts from a no-op core.
    """
    def __init__(self, hidden_dim: int, expansion: float = 2.0,
                 dropout: float = 0.0):
        super().__init__()
        inner = max(1, int(hidden_dim * expansion))
        self.fc1 = nn.Linear(hidden_dim, inner)
        self.fc2 = nn.Linear(inner, hidden_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return h + self.fc2(self.dropout(self.act(self.fc1(h))))


class Codebook(nn.Module):
    """Discrete-codebook core (straight-through estimator).

    Quantises h to the nearest entry in a learnable codebook of size K. Useful
    for instantiations like Token Assorted, where the standardisation is a
    discrete latent code rather than a token embedding or a continuous state.
    """
    def __init__(self, hidden_dim: int, num_codes: int = 512,
                 commit_weight: float = 0.25):
        super().__init__()
        self.codes = nn.Parameter(torch.randn(num_codes, hidden_dim) * 0.02)
        self.commit_weight = commit_weight
        self._last_commit_loss: torch.Tensor | None = None

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # (..., D) -> (..., K) distance to each code
        flat = h.reshape(-1, h.size(-1))                          # (N, D)
        d2 = (flat.pow(2).sum(-1, keepdim=True)                   # (N, 1)
              + self.codes.pow(2).sum(-1)                         # (K,)
              - 2 * flat @ self.codes.t())                        # (N, K)
        idx = d2.argmin(dim=-1)                                   # (N,)
        quant = self.codes[idx].view_as(h)                        # (..., D)
        # Commitment loss for the encoder side (h to be close to its code).
        self._last_commit_loss = self.commit_weight * (h - quant.detach()).pow(2).mean()
        # Straight-through: gradient flows as if quant == h.
        return h + (quant - h).detach()

    def pop_commit_loss(self) -> torch.Tensor | None:
        loss, self._last_commit_loss = self._last_commit_loss, None
        return loss
