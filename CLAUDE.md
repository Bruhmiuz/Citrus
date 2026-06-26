# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Research framework implementing a **reasoning-body standardisation module** over a frozen base LLM. The decomposition is:

```
N_in → R → S → R → S → ... → S → R → N_out
```

`R` = frozen middle transformer layers; `S = tail → core_S → head` = LoRA-adapted boundary layers + configurable core. Only LoRA deltas and `core_S` parameters are trainable; the base model is fully frozen.

## Running tests

```bash
# Structural test — no HuggingFace dependency, tests all configurations on a mock model
python examples/structural_test.py

# Smoke test — requires `pip install transformers`, downloads TinyLlama
python examples/smoke_test.py
```

There is no build step. Dependencies are PyTorch and (optionally) `transformers`.

## Architecture

All code lives in `reasoning_body/`.

**Public API:**

```python
ReasoningLoopModel(base, k_in, k_out, standardizations=[(head_cfg, tail_cfg, core_S), ...])
model(input_ids, schedule=[0, 1, 0, 2])   # explicit list of standardization indices
```

`k_in`/`k_out` are owned by the model. Standardizations share that boundary region; each picks its own LoRA pattern within it. There is no `num_loop_steps` and no single-S shorthand — `schedule` is required and `standardizations` is always a list.

**Data flow through `ReasoningLoopModel`:**
1. `encode()` — embeds tokens, runs `layers[0 : L - k_out]`
2. `loop_step(h, idx)` (once per schedule entry) — `standardizations[idx]` (= `tail → core_S → head`) then `layers[k_in : L - k_out]`
3. `decode()` — runs `layers[L - k_out : L]`, norm, lm_head

**Module responsibilities:**
- `config.py` — `BoundaryConfig(ranks, alpha, targets, init, train_base)`; constructor functions (`empty`, `frozen`, `uniform`, `inner_heavy`, `outer_heavy`, `skip_outer`) emit the rank list shape. `init` selects LoRA init (`"zero"` or `"soft_skip"`); `train_base` unfreezes the Boundary's base layer weights.
- `lora.py` — `LoRADelta` plus the `lora_active()` context manager that installs/removes forward hooks per Boundary call. Active only during loop steps.
- `boundary.py` — `Boundary` owns LoRA deltas but holds base-layer references in a plain Python list (`_base_layers`), not `nn.ModuleList`, so FSDP parameter discovery doesn't double-count them. `Boundary.forward` is `@torch.compiler.disable`d to make the hook-induced graph break explicit.
- `standardization.py` — `Standardization(base, head_cfg, tail_cfg, core_S, k_in, k_out)` bundles the per-S head/tail Boundaries and the core. One per entry in the model's standardizations list.
- `core.py` — `S` implementations: `Identity`, `ArgmaxResample` (CoT), `SoftmaxEmbedMixture` (Soft Thinking), `MLPCore`, `Codebook`.
- `configurations.py` — canonical `(head_cfg, tail_cfg, core_factory)` triples; compose into a list to pass as `standardizations`.
- `model.py` — `ReasoningLoopModel`; the only public-facing class.

**`BoundaryConfig.ranks` semantics** (innermost→outermost, i.e. ranks[0] is adjacent to the body):
- `ranks[i] > 0` → trainable LoRA adapter at that rank on the i-th inner slot
- `ranks[i] == 0` → base layer runs in the loop, frozen, no delta
- `i >= len(ranks)` → slot is **implicitly skipped** (layer bypassed in the loop)

A config's rank list may be shorter than the model's `k_in`/`k_out` — outer slots are skipped automatically. Optimal shape (inner-heavy vs outer-heavy) is an open empirical question and likely depends on `core_S`: inner-heavy fits vocab-grounded cores (Soft Thinking, ArgmaxResample); outer-heavy fits Identity / body-format cores where outer boundary layers need the most LoRA capacity to compensate for input-format mismatch.

**`BoundaryConfig.init`** is `"zero"` (default — Kaiming-A / zero-B, delta=0 at init) or `"soft_skip"` (SVD-init so that `scale·BA ≈ -W_topr` — the wrapped layer's contribution is partially cancelled at init). `"soft_skip"` is the natural companion to outer-heavy + Identity core.

**`BoundaryConfig.train_base`** (default `False`): when `True`, the Boundary calls `requires_grad_(True)` on every parameter of every base layer it holds — both `ranks[i] == 0` and `ranks[i] > 0` slots. The body (layers between `k_in` and `L − k_out`) stays frozen regardless. This is the path to CoT-style fine-tuning of the boundary region. Note: base layers are shared across Standardizations — if any one Boundary over a layer sets `train_base=True`, that layer is trainable for every Standardization.

**Integration notes (KV cache, FSDP, torch.compile):** see `README.md` and the docstring on `ReasoningLoopModel`. Short version: `**kw` is threaded to every layer call, but don't pass `past_key_values` to a `forward()` with a non-empty schedule (loop re-passes corrupt the cache). FSDP wrap plan targets each `Standardization` and each `base.model.layers[i]`. Each `Boundary.forward` is a graph break under torch.compile; `encode`/`decode` compile cleanly.

## Literature

`reasoning_body_framework_papers.md` is the supporting bibliography. Papers are organised into six categories:

1. **Layer specialisation** (§1, the empirical backbone) — five convergent angles on the boundary/body partition:
   - 1a multilingual stratification (Wendler, Wu, Zhao, Dumas, Fierro) — Dumas adds causal evidence via activation patching; Fierro factorises factual recall into body subject-enrichment vs late object-extraction
   - 1b detokenisation / vocabulary projection (Lad, Belrose, Geva)
   - 1c middle-layer redundancy (Gromov)
   - 1d probing (Skean)
   - 1e cross-architecture / cross-modality convergence (Huh — Platonic Representation Hypothesis)
2. **LLM Neuroanatomy / RYS** (§2) — independent grey-literature validation (David Noel Ng's RYS I–III, 2026): duplicating middle layers without weight changes tops the HF Open LLM Leaderboard; multilingual probing across five frontier models confirms the three-phase decode/reason/encode partition with a ~15-layer boundary constant.
3. **Standardisation function instances** (§3) — COCONUT, Soft Thinking, Saunshi (looped transformers), plus newer entries: Deng (Chain of Superposition theory), Lei (Latent-SFT vocab-column-space), Tack (CoCoMix SAE-dictionary), Wang (Planning Tokens — thinnest trainable S).
4. **Related latent-reasoning methods** (§4, to be mapped) — CODI, HybridCoT, Zhu (Reasoning by Superposition).
5. **Mechanistic interpretability foundations** (§5) — nostalgebraist (Logit Lens, the historical anchor for layer-wise vocabulary readouts) and Elhage (Toy Models of Superposition — geometric foundation for the chain-of-superposition framing).
6. **Surveys** (§6) — implicit reasoning survey.

## Development guidelines

- **Ask, don't assume.** If something is unclear, ask before writing a single line. Never make silent assumptions about intent, architecture, or requirements.
- **Simplest solution first.** Implement the simplest thing that works. Do not add abstractions or flexibility that weren't explicitly requested.
- **Don't touch unrelated code.** If a file or function is not directly part of the current task, do not modify it, even if it could be improved.
- **Flag uncertainty explicitly.** If you are not confident about an approach, say so before proceeding.
- Suggestions for better approaches are welcome, especially ones with lasting architectural impact over tactical fixes.
