# Reasoning Body Framework

A plug-and-play standardisation module over a frozen base LLM, implementing the decomposition:

```
N_in → R → S → R → S → ... → S → R → N_out
```

where `R` is the reasoning body (frozen middle transformer layers) and each `S = tail → core_S → head` is a standardisation module (LoRA-adapted base boundary layers + a configurable core). A `ReasoningLoopModel` can hold multiple `S` and the caller supplies a `schedule` selecting which to apply at each loop step.

## Layout

```
reasoning_body/
├── config.py            BoundaryConfig (ranks list) + gradient-shape constructors
├── lora.py              LoRADelta + hook-based activation
├── boundary.py          The Boundary module (per-layer LoRA / frozen / implicit-skip)
├── standardization.py   Standardization = tail + core_S + head (one per S)
├── core.py              Identity, ExactNextToken, SoftmaxEmbedMixture, MLPCore, Codebook
├── model.py             ReasoningLoopModel — encode / loop_step / decode pipeline
├── configurations.py    Canonical configurations (COCONUT, CoT, Soft Thinking, novel)
└── __init__.py
examples/
├── structural_test.py   Runs every configuration on a mock model (no HF dependency)
└── smoke_test.py        Runs every configuration on a real HF model (TinyLlama by default)
```

## Indexing convention

`BoundaryConfig.ranks` is a list of LoRA ranks indexed **innermost (adjacent to body) → outermost (adjacent to embedding/unembedding)**, identically for head and tail. This lets you reason about the rank gradient as a single shape: rank-decreasing-with-distance-from-body is the framework-predicted shape, rank-decreasing-toward-body is the counter-prediction.

A rank list shorter than the model's `k_in`/`k_out` means the outermost slots are **implicitly skipped** (bypassed entirely during the loop). Per slot:

- `ranks[i] > 0` → trainable LoRA adapter at that rank
- `ranks[i] == 0` → base layer runs in the loop, frozen, no delta
- `i >= len(ranks)` → layer is bypassed during the loop pass

## BoundaryConfig options

`BoundaryConfig(ranks, alpha=16.0, targets=DEFAULT_LORA_TARGETS, init="zero", train_base=False)`. The two non-obvious options:

**`init`** — how each LoRA delta is initialised. Per-Boundary; applies to every `ranks[i] > 0` slot.

| value | initial state of the wrapped projection |
|---|---|
| `"zero"` (default) | `A ~ Kaiming, B = 0` → delta = 0 → wrapped layer ≡ base layer at init |
| `"soft_skip"` | `scale·BA ≈ -W_topr` (rank-r truncated SVD of negated base weight) → wrapped layer ≈ `(W − W_topr)x` at init, partially neutralising the layer's contribution |

`"soft_skip"` is the natural pair for outer-heavy + Identity core: the outer boundary layers are doing vocab-projection work that an Identity core does not want, and starting the LoRA close to "cancel it" is a shorter training path than starting at zero. For inner-heavy + vocab-grounded cores (Soft Thinking, ExactNextToken), `"zero"` is fine — the boundary's natural action is already wanted, LoRA just refines it.

**`train_base`** — when `True`, the Boundary unfreezes the parameters of every base layer it holds (both `ranks[i] == 0` slots and `ranks[i] > 0` slots). The base model body — the layers between `k_in` and `L − k_out` — stays frozen regardless. Use this to replicate CoT-style fine-tuning of the boundary region:

```python
# CoT-style fine-tuning of the boundary, no LoRA, body kept frozen.
head_cfg, tail_cfg, _ = cfgs.frozen_baseline(k_in=4, k_out=4, train_base=True)
model = ReasoningLoopModel(
    base, k_in=4, k_out=4,
    standardizations=[(head_cfg, tail_cfg, ExactNextToken(base.model.norm, base.lm_head, base.model.embed_tokens))],
)
```

Base layers are shared across Standardizations. If any one Boundary over a given layer sets `train_base=True`, that layer's parameters are trainable for every Standardization (a single `requires_grad` flag per parameter).

## Minimal usage

```python
from transformers import AutoModelForCausalLM
from reasoning_body import ReasoningLoopModel
from reasoning_body import configurations as cfgs

base = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")

# Inner-heavy LoRA, identity core. k_in = k_out = 4.
head_cfg, tail_cfg, core_factory = cfgs.inner_heavy_lora(k_in=4, k_out=4, max_rank=16)
model = ReasoningLoopModel(
    base, k_in=4, k_out=4,
    standardizations=[(head_cfg, tail_cfg, core_factory(base))],
)

# Only LoRA deltas (and core_S parameters, if any) are trainable.
print(model.trainable_parameter_count())

# Forward with schedule of length 2 — apply the (only) standardization twice.
logits = model(input_ids, schedule=[0, 0])
```

## Mixing standardizations

The model takes a list; the schedule picks which to apply per step:

```python
h1, t1, c1 = cfgs.inner_heavy_lora(k_in=4, k_out=4, max_rank=16)
h2, t2, c2 = cfgs.soft_thinking(temperature=0.8)

model = ReasoningLoopModel(
    base, k_in=4, k_out=4,
    standardizations=[
        (h1, t1, c1(base)),   # S0: inner-heavy LoRA + identity core
        (h2, t2, c2(base)),   # S1: no boundary + soft-mixture core
    ],
)

# Alternate inner-heavy with soft-mixture across 4 loop steps.
logits = model(input_ids, schedule=[0, 1, 0, 1])
```

Each standardization shares the model's `k_in`/`k_out` but independently chooses how to populate that boundary region. `len(head_cfg.ranks)` may be anywhere from `0` to `k_in` (outer slots skipped), and likewise for the tail.

## Custom configurations

A configuration is a triple `(head_cfg, tail_cfg, core_factory)`. Build one by hand:

```python
from reasoning_body import BoundaryConfig
from reasoning_body.core import SoftmaxEmbedMixture

# k_in = 4, but only the 2 innermost slots are active; outermost 2 are skipped.
# Outer-heavy + soft_skip init: cancel the outer layers' contribution at init.
head_cfg = BoundaryConfig(ranks=[8, 16], init="soft_skip")  # ranks[1] outermost-of-active

# k_out = 4, all 4 slots active; mix of LoRA ranks and one frozen slot.
# train_base=True: the boundary's base weights are also trainable here.
tail_cfg = BoundaryConfig(ranks=[16, 8, 2, 0], train_base=True)  # ranks[3]=0 → frozen+trainable

# core_S is any nn.Module: H -> H.
core = SoftmaxEmbedMixture(base.lm_head, base.model.embed_tokens, temperature=0.8)

model = ReasoningLoopModel(
    base, k_in=4, k_out=4,
    standardizations=[(head_cfg, tail_cfg, core)],
)
```

## Freeze modes

`model.set_freeze_mode(mode)` toggles which of the two trainable-eligible components carries gradients. It rewrites `requires_grad` on exactly two parameter groups and records the choice on `model.freeze_mode`:

- **reasoning** — the body layers `layers[k_in : L − k_out]` (`model.reasoning_parameters()`).
- **standardization** — the LoRA adapters plus each `core_S`'s *own* parameters (`model.standardization_parameters()`). Base parameters a `core_S` merely references — `embed` / `norm` / `lm_head` in the vocabulary-grounded cores — are excluded; they stay frozen with the base.

| mode | reasoning body | standardization |
|---|---|---|
| `"reasoning"` (default) | frozen | trainable |
| `"standardization"` | trainable | frozen |
| `"none"` | trainable | trainable |

```python
model.set_freeze_mode("standardization")   # train the body, freeze adapters + cores
model.set_freeze_mode("none")              # train both
print(model.freeze_mode)                    # -> "none"
```

The construction-time default is `"reasoning"` — the same frozen-body / trainable-standardization partition the base freeze already produces.

**What the toggle never unfreezes:** the boundary base layers (first `k_in`, last `k_out`), `embed_tokens`, `norm`, and `lm_head` sit outside both groups and stay frozen (governed only by construction and `BoundaryConfig.train_base`). So `"none"` frees *all* transformer layers only when `k_in = k_out = 0`; with a non-empty boundary region those layers remain frozen. `set_freeze_mode` and `train_base` are orthogonal — the latter is the only path to training the boundary layers.

## Method mapping

`ranks=[a, b, c, ...]` is innermost → outermost. A shorter list means the outer slots are implicitly skipped.

| Method                  | k_in | k_out | head ranks            | tail ranks            | core_S               |
|-------------------------|------|-------|-----------------------|-----------------------|----------------------|
| COCONUT                 | 0    | 0     | —                     | —                     | Identity             |
| CoT                     | 0    | 0     | —                     | —                     | ExactNextToken       |
| Soft Thinking           | 0    | 0     | —                     | —                     | SoftmaxEmbedMixture  |
| Frozen-boundary baseline| k    | k     | `[0]*k`               | `[0]*k`               | Identity             |
| Uniform LoRA            | k    | k     | `[R]*k`               | `[R]*k`               | Identity             |
| Inner-heavy LoRA        | 4    | 4     | `[16, 8, 4, 2]`       | `[16, 8, 4, 2]`       | Identity             |
| Outer-heavy LoRA        | 4    | 4     | `[2, 4, 8, 16]`       | `[2, 4, 8, 16]`       | Identity             |
| Skip-outer + LoRA-inner | 4    | 4     | `[8, 8]` (outer skipped) | `[8, 8]` (outer skipped) | Identity             |

Which shape is optimal is an empirical question and likely depends on `core_S` — inner-heavy is the natural fit for vocab-grounded cores (Soft Thinking, ExactNextToken), outer-heavy for Identity / body-format cores. See the framework notes for the argument.

`ExactNextToken` re-embeds the model's exact greedy next-token prediction: it runs the final `norm` before the unembedding (`argmax(lm_head(norm(h)))`), exactly as `decode` does, so the re-embedded token is the base model's true prediction rather than an un-normed approximation. Construct it as `ExactNextToken(base.model.norm, base.lm_head, base.model.embed_tokens)` (pass `norm=None` for a base model with no final norm).

## Notes on FSDP

Each `Boundary` holds base-layer references in a plain Python list (`_base_layers`), not an `nn.ModuleList`. PyTorch parameter discovery does not traverse plain lists, so even with `N` standardizations sharing the same `k_in`/`k_out` boundary region, each base layer's parameters are owned exclusively by `self.base`:

- `model.parameters()` yields the base model's parameters once, plus per-standardization LoRA deltas and `core_S` parameters. No double-counting under multi-S.
- The `fully_shard` plan should target each `base.model.layers[i]` and each `model.standardizations[j]` independently. Wrapping at the `Standardization` level is the natural unit (it owns its tail, head, and core_S).
- The base model is frozen via `requires_grad_(False)` at construction time. Any Boundary built with `train_base=True` then re-enables `requires_grad` on its base layers, so those specific layers participate in training while the body remains frozen.

## Notes on attention / KV cache

`_call_layer` and every stage method (`encode`, `loop_step`, `decode`, `forward`) thread `**layer_kwargs` through to each base-layer call, so you can pass `attention_mask`, `position_ids`, `cache_position`, etc. The framework does not own or update those values.

**KV cache caveat for loops:** do not pass `past_key_values` or `use_cache=True` to `forward(input_ids, schedule=…)` when `schedule` is non-empty. Each loop step re-processes the same sequence positions through the body; appending to a cache per iteration corrupts it. For autoregressive generation, manage the cache externally and call `forward` repeatedly on growing inputs (or call `encode`/`decode` directly when you need stage-level control).

## Notes on torch.compile

`Boundary.forward` is marked `@torch.compiler.disable` because the LoRA activation pattern (`lora_active` registers and removes forward hooks per call) cannot be traced by TorchDynamo. This produces a well-defined graph break at each `Boundary` call rather than an unpredictable dynamo bail.

In practice:

- `encode` and `decode` compile cleanly end-to-end (pure layer iteration, no hooks).
- `loop_step` compiles the body iteration and the dispatch to `standardizations[idx]`, with a graph break inside each `Boundary.forward`. The body-loop graph is preserved.
- The number of compiled graphs scales with the number of distinct `Standardization` calls in a `schedule`, not with `len(schedule)` (dynamo caches per traced path).
