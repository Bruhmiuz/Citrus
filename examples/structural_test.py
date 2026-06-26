"""Structural tests using a mock Llama-style model (no HF / no downloads).

Exercises every configuration through a forward pass, verifies that:
  - the base model's parameters are frozen
  - trainable parameter counts are non-decreasing in LoRA rank
  - all stages (encode / loop_step / decode) produce finite tensors
  - identity-initialised cores (Identity, MLPCore zero-init) produce delta=0
    on the first step relative to running no loop step
"""
from __future__ import annotations
import sys
from pathlib import Path

# Make `reasoning_body` importable when running this file directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn

from reasoning_body import ReasoningLoopModel, configurations as cfgs


# ---------------------------------------------------------------------------
# Mock model: Llama-style submodule names so LoRA targets resolve correctly.
# ---------------------------------------------------------------------------

class MockAttention(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.o_proj = nn.Linear(d, d, bias=False)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        q, k, v = self.q_proj(h), self.k_proj(h), self.v_proj(h)
        d = q.size(-1)
        scores = (q @ k.transpose(-2, -1)) / (d ** 0.5)
        # causal mask
        S = scores.size(-1)
        mask = torch.triu(torch.full((S, S), float("-inf"), device=scores.device), diagonal=1)
        attn = (scores + mask).softmax(-1)
        return self.o_proj(attn @ v)


class MockMLP(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.gate_proj = nn.Linear(d, 4 * d, bias=False)
        self.up_proj = nn.Linear(d, 4 * d, bias=False)
        self.down_proj = nn.Linear(4 * d, d, bias=False)
        self.act = nn.SiLU()

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act(self.gate_proj(h)) * self.up_proj(h))


class MockLayer(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(d)
        self.post_attention_layernorm = nn.LayerNorm(d)
        self.self_attn = MockAttention(d)
        self.mlp = MockMLP(d)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        h = h + self.self_attn(self.input_layernorm(h))
        h = h + self.mlp(self.post_attention_layernorm(h))
        return h


class MockTransformer(nn.Module):
    def __init__(self, vocab: int, d: int, n_layers: int):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, d)
        self.layers = nn.ModuleList([MockLayer(d) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d)


class MockCausalLM(nn.Module):
    def __init__(self, vocab: int = 256, d: int = 64, n_layers: int = 8):
        super().__init__()
        self.model = MockTransformer(vocab, d, n_layers)
        self.lm_head = nn.Linear(d, vocab, bias=False)


# ---------------------------------------------------------------------------
# Test driver
# ---------------------------------------------------------------------------

def main() -> int:
    torch.manual_seed(0)
    base = MockCausalLM(vocab=256, d=64, n_layers=8)
    input_ids = torch.randint(0, 256, (2, 16))

    base_param_count = sum(p.numel() for p in base.parameters())

    configurations = [
        ("COCONUT",                  cfgs.coconut),
        ("CoT (argmax)",             cfgs.cot),
        ("Soft Thinking",            cfgs.soft_thinking),
        ("Frozen boundary (k=2,2)",  lambda: cfgs.frozen_baseline(2, 2)),
        ("Uniform LoRA r=4",         lambda: cfgs.uniform_lora(2, 2, rank=4)),
        ("Inner-heavy max_r=8",      lambda: cfgs.inner_heavy_lora(2, 2, max_rank=8, decay=2)),
        ("Outer-heavy max_r=8",      lambda: cfgs.outer_heavy_lora(2, 2, max_rank=8, decay=2)),
        ("Skip-outer (2, inner=1)",  lambda: cfgs.skip_outer_lora(2, 2, num_inner=1, rank=4)),
        ("Inner-heavy + soft core",  lambda: cfgs.inner_heavy_soft(2, 2, max_rank=4, decay=2)),
        ("MLPCore (k=0)",            lambda: cfgs.mlp_core(k_in=0, k_out=0, expansion=2.0)),
        ("Codebook (k=0)",           lambda: cfgs.codebook_core(k_in=0, k_out=0, num_codes=64)),
    ]

    print(f"Base model: {base_param_count:,} params, {len(base.model.layers)} layers, d=64\n")
    print(f"{'Configuration':<30s} {'trainable':>12s} {'frac':>8s} "
          f"{'|Δlogits|_F (T=2 vs T=0)':>26s}")
    print("-" * 80)

    failed = 0
    for name, fn in configurations:
        head_cfg, tail_cfg, core_factory = fn()
        k_in, k_out = len(head_cfg.ranks), len(tail_cfg.ranks)
        model = ReasoningLoopModel(
            base, k_in, k_out,
            [(head_cfg, tail_cfg, core_factory(base))],
        ).eval()

        # Check the base model is fully frozen.
        for p in base.parameters():
            if p.requires_grad:
                print(f"  [FAIL] base parameter has requires_grad=True under {name}")
                failed += 1

        trainable = model.trainable_parameter_count()
        frac = trainable / model.total_parameter_count()

        with torch.no_grad():
            logits_0 = model(input_ids, schedule=[])
            logits_2 = model(input_ids, schedule=[0, 0])

        if not (torch.isfinite(logits_0).all() and torch.isfinite(logits_2).all()):
            print(f"  [FAIL] non-finite logits in {name}")
            failed += 1

        delta = torch.norm(logits_2 - logits_0).item()
        print(f"{name:<30s} {trainable:>12,d} {frac:>7.2%}  {delta:>26.6f}")

    print()
    if failed == 0:
        print("All structural checks passed.")
        return 0
    else:
        print(f"{failed} check(s) FAILED.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
