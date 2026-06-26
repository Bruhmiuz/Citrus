"""Smoke test on a real HuggingFace causal LM.

Requires: `pip install transformers`

Defaults to TinyLlama for quick iteration. Override MODEL_NAME for larger models
of the same architecture family (any Llama-style HF model with `.model.layers`,
`.model.embed_tokens`, `.model.norm`, `.lm_head`).
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    print("transformers not installed; this test requires `pip install transformers`")
    sys.exit(1)

from reasoning_body import ReasoningLoopModel, configurations as cfgs


MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DTYPE = torch.bfloat16


def build_and_count(base, fn):
    head_cfg, tail_cfg, core_factory = fn()
    k_in, k_out = len(head_cfg.ranks), len(tail_cfg.ranks)
    model = ReasoningLoopModel(
        base, k_in, k_out,
        [(head_cfg, tail_cfg, core_factory(base))],
    ).eval()
    return model


def main():
    print(f"Loading {MODEL_NAME} (dtype={DTYPE}) ...")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    base = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=DTYPE)
    base.eval()

    prompt = "The capital of France is"
    input_ids = tok(prompt, return_tensors="pt").input_ids

    L = len(base.model.layers)
    base_total = sum(p.numel() for p in base.parameters())
    print(f"Base model: L={L} layers, {base_total:,} parameters\n")

    configurations = [
        ("COCONUT",                  cfgs.coconut),
        ("CoT (argmax)",             cfgs.cot),
        ("Soft Thinking",            cfgs.soft_thinking),
        ("Frozen boundary k=4,4",    lambda: cfgs.frozen_baseline(4, 4)),
        ("Uniform LoRA r=8",         lambda: cfgs.uniform_lora(4, 4, rank=8)),
        ("Inner-heavy max_r=16",     lambda: cfgs.inner_heavy_lora(4, 4, max_rank=16, decay=2)),
        ("Skip-outer inner=2",       lambda: cfgs.skip_outer_lora(4, 4, num_inner=2, rank=8)),
        ("Inner-heavy + soft core",  lambda: cfgs.inner_heavy_soft(4, 4, max_rank=8, decay=2)),
    ]

    print(f"{'Configuration':<28s} {'trainable':>12s} {'frac':>7s} {'top-1 (T=0)':>16s} {'top-1 (T=2)':>16s}")
    print("-" * 85)

    for name, fn in configurations:
        model = build_and_count(base, fn)
        trainable = model.trainable_parameter_count()
        frac = trainable / model.total_parameter_count()

        with torch.no_grad():
            logits_0 = model(input_ids, schedule=[])
            logits_2 = model(input_ids, schedule=[0, 0])

        top1_0 = tok.decode(logits_0[0, -1].argmax().item())
        top1_2 = tok.decode(logits_2[0, -1].argmax().item())

        print(f"{name:<28s} {trainable:>12,d} {frac:>6.2%}  "
              f"{top1_0!r:>16s} {top1_2!r:>16s}")


if __name__ == "__main__":
    main()
