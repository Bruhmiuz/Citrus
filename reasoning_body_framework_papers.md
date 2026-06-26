# Reasoning Body Framework — Supporting Literature

A working bibliography for the framing, motivation, and empirical grounding of the *reasoning body* / *standardisation module* decomposition. Papers are grouped by the role they play in the argument.

---

## 1. Layer Specialisation: Why Boundary Layers Belong to the Standardisation Module

This is the central evidentiary category for the framework. The claim is that the first $k_\text{in}$ and last $k_\text{out}$ transformer layers perform *interface* operations (tokenisation/surface-form ↔ semantic/abstract representation), while middle layers perform iterative computation on a shared, language-agnostic representational format. The papers below support this from five convergent angles: (a) multilingual stratification, (b) detokenisation / vocabulary projection, (c) middle-layer redundancy, (d) probing evidence for richer middle-layer features, and (e) cross-architecture / cross-modality convergence.

### 1a. Multilingual stratification — surface ↔ language-agnostic ↔ surface

**Wendler et al. 2024 — Do Llamas Work in English? On the Latent Language of Multilingual Transformers**
*arXiv:* [2402.10588](https://arxiv.org/abs/2402.10588) · ACL 2024
Tracks intermediate embeddings in Llama-2 through three distinct phases: (1) far from any output-token region (input space, early layers), (2) decodable to a semantically correct next token, with higher probability given to its English form than the input-language form (concept space, middle layers), (3) movement into the input-language-specific region (output space, late layers). The "concept space" lies closer to English than to other languages. **Direct support for the partition**: early layers handle surface form, middle layers carry a language-neutral semantic representation, late layers re-introduce target-language surface form.

**Wu et al. 2024 — The Semantic Hub Hypothesis: Language Models Share Semantic Representations Across Languages and Modalities**
*arXiv:* [2411.04986](https://arxiv.org/abs/2411.04986)
Argues that LMs learn a shared representation space (a "semantic hub", analogous to the neuroscience hub-and-spoke model) where semantically equivalent inputs across languages, code, arithmetic, vision, and audio are placed near one another in the **intermediate layers**. Demonstrates that interventions in this shared space in one modality affect outputs in another, indicating the hub is actively used, not vestigial. **Strengthens Wendler**: the layer-stratified pattern is not specific to language but generalises to modality, which is exactly the kind of cross-domain robustness the reasoning body framework predicts.

**Zhao et al. NeurIPS 2025 — When Less Language is More: Language-Reasoning Disentanglement Makes LLMs Better Multilingual Reasoners**
*OpenReview:* [language-reasoning disentanglement](https://openreview.net/pdf/dc1102642c3731185a62af150f47222909c3624e.pdf) · NeurIPS 2025 poster
Identifies a language-specific subspace $M_s$ inside transformer hidden states via SVD; ablating it at inference time consistently improves multilingual reasoning across 10 open-weight LLMs and 11 typologically diverse languages. Crucially, **the optimal layers at which to intervene are language-specific and concentrated in middle layers**, confirming that reasoning content and language content are separable along both subspace and depth axes. **Most direct evidence** for the existence of a language-agnostic computational substrate that can be isolated from surface-form layers.

**Dumas, Wendler, Veselovsky, Monea & West 2024 — Separating Tongue from Thought**
*arXiv:* [2411.08745](https://arxiv.org/abs/2411.08745)
Activation-patching experiments on Llama 2 7B that swap the *concept* while holding language fixed, and vice versa. Mean-across-language concept vectors improve translation when patched in. **Cleanest causal evidence for the separability of language and concept at the hidden-state level**: not just correlational geometry (Wendler, Wu) but interventional. Directly supports the claim that boundary layers do language-surface work while the body carries language-agnostic content.

**Fierro, Foroutan, Elliott & Søgaard 2025 — How Do Multilingual Language Models Remember Facts?**
*ACL Findings 2025:* [aclanthology.org/2025.findings-acl.827](https://aclanthology.org/2025.findings-acl.827/)
Decomposes factual recall into a **language-independent subject-enrichment stage (middle layers)** and a **language-specific object-extraction stage (late layers)**. A direct functional factorisation that maps onto the encode/reason/decode partition: facts are stored and manipulated in a language-agnostic body, then projected into a target-language vocabulary by N_out.

### 1b. Detokenisation, vocabulary projection, and the "stages of inference"

**Lad, Gurnee & Tegmark 2024 — The Remarkable Robustness of LLMs: Stages of Inference?**
*arXiv:* [2406.19384](https://arxiv.org/abs/2406.19384)
Deleting and swapping adjacent layers retains 72–95% of original prediction accuracy without fine-tuning. The localized sensitivity pattern motivates a four-stage hypothesis observed across eight models: (1) **detokenisation** (early), lifting raw token embeddings into contextual representations; (2) **feature engineering** (middle), iterative refinement of task/entity features; (3) **prediction ensembling** (later), aggregating hidden states toward plausible next tokens; (4) **residual sharpening** (final), suppressing irrelevant features. **The two boundary stages (1 and 4) map exactly onto $\mathcal{N}_\text{in}$ and $\mathcal{N}_\text{out}$ in the framework**, while stages 2–3 correspond to the reasoning body $\mathcal{R}$.

**Belrose et al. 2023 — Eliciting Latent Predictions from Transformers with the Tuned Lens**
*arXiv:* [2303.08112](https://arxiv.org/abs/2303.08112)
Trains an affine probe per block to decode each hidden state into a vocabulary distribution. Shows that mid-stack residuals are not yet aligned with the output distribution, and that alignment grows rapidly in the final layers — i.e. **late layers do the work of projecting toward vocabulary space**, an interface operation rather than a computational one. Refines the earlier "logit lens" (nostalgebraist, see §5) and shows the trajectory of latent predictions can be used causally.

**Geva et al. 2021 — Transformer Feed-Forward Layers Are Key-Value Memories**
*arXiv:* [2012.14913](https://arxiv.org/abs/2012.14913) · EMNLP 2021
FFN layers act as key-value stores. Lower layers capture **shallow patterns**; upper layers capture **semantic patterns** and induce output distributions that concentrate mass on tokens likely to follow. Demonstrates that the late-layer FFN behaviour is specifically vocabulary-promotion. **Mechanistic confirmation** that the late layers are functionally specialised for output-space projection.

### 1c. Middle-layer redundancy / boundary-layer fragility

**Gromov et al. 2024 — The Unreasonable Ineffectiveness of the Deeper Layers**
*arXiv:* [2403.17887](https://arxiv.org/abs/2403.17887) · ICLR 2025
Up to half of the layers can be pruned (selecting by inter-layer similarity, with a small QLoRA "heal" step) before significant degradation, but the pruning is concentrated in **middle-to-deep middle** layers — early and final layers are not interchangeable. **Supports the framework's structural claim**: middle layers operate on a shared representational format and are individually fungible, while boundary layers do irreplaceable interface work. Also justifies the practical choice to **freeze the reasoning body** in a plug-and-play setup — middle layers are robust to perturbation as long as the input format is preserved.

### 1d. Probing evidence: middle layers carry richer features

**Skean et al. ICML 2025 — Layer by Layer: Uncovering Hidden Representations in Language Models**
*arXiv:* [2502.02013](https://arxiv.org/abs/2502.02013)
Across 32 text-embedding tasks and multiple architectures (transformers and state-space models), **intermediate-layer representations consistently outperform final-layer representations** on downstream tasks. Proposes an information-theoretic / geometric framework explaining the trade-off between compression and signal preservation. **Quantitative support** that the body-output hidden state (the input/output to $S$) is the natural locus for cross-task generalisation — exactly what Stage 2 probing intends to exploit.

### 1e. Cross-architecture / cross-modality representational convergence

**Huh, Cheung, Wang & Isola 2024 — The Platonic Representation Hypothesis**
*arXiv:* [2405.07987](https://arxiv.org/abs/2405.07987)
Argues that models trained on sufficient diverse data converge — across architectures and modalities — on a shared statistical representation of the underlying world. The language-agnostic middle (§1a) is a special case of this claim along the multilingual axis. **Broadest theoretical umbrella over the layer-stratification evidence**: if convergence to a shared semantic substrate is the generic asymptote of large-scale training, then the body's role as a model-agnostic computational format is not incidental but structural — the framework's plug-and-play standardisation assumption is on principled ground.

---

## 2. LLM Neuroanatomy — Independent Empirical Validation (RYS Series)

Grey literature with hard public anchoring: independently arrives at the same three-phase decomposition the framework formalises, with leaderboard-verifiable evidence.

**David Noel Ng 2026 — LLM Neuroanatomy I: How I Topped the LLM Leaderboard Without Changing a Single Weight**
[dnhkng.github.io/posts/rys/](https://dnhkng.github.io/posts/rys/) · published 2026-03-10
Introduces **RYS (Repeat Yourself)**: duplicating a contiguous block of middle layers in Qwen2-72B with **no weight modification** reaches #1 on the HuggingFace Open LLM Leaderboard (IFEval, BBH, MATH Lvl 5, GPQA, MuSR, MMLU-PRO). Core finding: duplication only works for **middle** layers — duplicating early or late layers breaks the model. Motivating observation: LLMs can reason coherently in Base64, implying early layers translate surface → abstract and late layers translate abstract → surface, with middle layers doing format-agnostic reasoning. **Independently arrives at the same three-phase anatomy (decode / reason / encode) the framework formalises**, with the publicly verifiable leaderboard result as hard empirical anchoring.

**David Noel Ng 2026 — LLM Neuroanatomy II (RYS Part II)**
[dnhkng.github.io/posts/rys-ii/](https://dnhkng.github.io/posts/rys-ii/)
Confirms RYS generalises across model sizes. Preliminary experiment (building on Evan Maunder) showing middle layers organise by **topic** rather than language. Sets up the Part III probing study.

**David Noel Ng 2026 — LLM Neuroanatomy III: Why RYS Works — The Language-Agnostic Middle**
[dnhkng.github.io/posts/sapir-whorf/](https://dnhkng.github.io/posts/sapir-whorf/) · published 2026-03-26, updated 2026-06-25
Scales the multilingual probing experiment to **five frontier models including MoEs** (Qwen3.5-27B, MiniMax M2.5, GLM-4.7, Gemma-4 31B, GPT-OSS-120B): 64 sentences (8 languages × 8 topics), pairwise cosine similarity at every layer. Finds the same **three-phase transition across all models**: early layers cluster by language, middle layers cluster by topic, late layers return to language clustering. Extends to **code and LaTeX with single-letter variables** (near-zero lexical overlap with English prose). **Encode/decode blocks are roughly constant at ~15 layers regardless of total model depth**; the reasoning block absorbs the remaining stack. Key framing: *"the anatomy predicts the surgery"* — the layers where the shared semantic space lives are exactly the layers RYS can profitably duplicate. Discusses Sapir-Whorf as the linguistic analogue. References Wendler 2024, Wu 2024, Dumas 2024, Fierro 2025, and the Platonic Representation Hypothesis.

**Framework implications from the RYS series:**

- **Independent triangulation.** Operationally validated three-phase decomposition that matches the framework's $\mathcal{N}_\text{in}$ / $\mathcal{R}$ / $\mathcal{N}_\text{out}$ partition.
- **Concrete prior on $k_\text{in}, k_\text{out}$.** The empirical ~15-layer boundary at frontier scale is a usable prior — the reasoning body absorbs the rest of the stack.
- **Behavioural signature of the partition.** *"Duplication improves, modification breaks"* is a clean operational test: body layers do iterative refinement on a stable format (duplication adds iterations), boundary layers do type-translation (duplication produces a wrong-type intermediate).
- **Limitation surfaced.** RYS with middle-layer sub-block duplication is **outside the framework's current formulation** — the body $\mathcal{R}$ is treated as monolithic. There may be exploitable structure *within* the body, worth flagging as a Stage 1 limitation or future direction.

---

## 3. Standardisation Function Instances (Existing Methods Re-framed)

Methods that, under the reasoning body framework, correspond to particular choices of $(k_\text{in}, k_\text{out}, S)$.

**Hao et al. 2024 — Training Large Language Models to Reason in a Continuous Latent Space (COCONUT)**
*arXiv:* [2412.06769](https://arxiv.org/abs/2412.06769) · COLM 2025 · [Code](https://github.com/facebookresearch/coconut)
Feeds the last hidden state back as the next input embedding, bypassing tokenisation. Reports emergent BFS-like reasoning patterns and outperforms CoT on tasks requiring backtracking (ProsQA, ProntoQA), with measurable gap on GSM8K. **Framework instantiation**: $k_\text{in} = k_\text{out} = 0$, $S = \mathcal{I}$ (identity).

**Zhang et al. NeurIPS 2025 — Soft Thinking: Unlocking the Reasoning Potential of LLMs in Continuous Concept Space**
*arXiv:* [2505.15778](https://arxiv.org/abs/2505.15778) · [Code](https://github.com/eric-ai-lab/Soft-Thinking)
Training-free. Replaces the sampled token at each step with a probability-weighted mixture of token embeddings, $S(h) = \sum_v p_v(h) \cdot e_v$. Stays vocabulary-grounded but avoids discrete commitment. **Framework instantiation**: $k_\text{in}, k_\text{out} > 0$ (full $\mathcal{N}_\text{in/out}$), $S$ = embed-by-softmax-mixture. Sits strictly between $S_\mathcal{N}$ and $S_\mathcal{I}$ on the vocab-grounding axis.

**Saunshi et al. ICLR 2025 — Reasoning with Latent Thoughts: On the Power of Looped Transformers**
*arXiv:* [2502.17416](https://arxiv.org/abs/2502.17416)
A $k$-layer transformer looped $L$ times nearly matches a $kL$-layer non-looped model on synthetic reasoning (addition, $p$-hop induction, math), with significantly higher reasoning-per-parameter inductive bias than perplexity would predict. **Architectural cousin** of the framework: the loop is over a fixed body, and the "standardisation" between iterations is implicit (no explicit boundary operation, just residual stream continuation). Useful for arguing that *depth-as-iteration* is a separate axis from *parameter count*.

**Deng et al. 2025 — LLM Latent Reasoning as Chain of Superposition**
*arXiv:* [2510.15522](https://arxiv.org/abs/2510.15522)
Proposes that latent reasoning hidden states are not singular interpretations but **superpositions of multiple reasoning chains simultaneously**. Each latent step occupies a superposition state rather than committing to one path; the model effectively runs BFS-like exploration in hidden-state space. **Framework instantiation**: explains why $S_\mathcal{I}$ (identity core) with repeated body iterations is more powerful than it appears — each pass refines a superposition, not a single hypothesis. Directly connects to Stage 3's consistency loss: if the body output is a superposition, the contrastive objective is asking structurally similar problems to produce consistent superpositions.

**Lei et al. 2025 — Latent-SFT: Latent Reasoning in LLMs as a Vocabulary-Space Superposition**
*OpenReview:* [pdf/d8d4c2aacdb0b4358bb3658ca5c98d25be8892ab](https://openreview.net/pdf/d8d4c2aacdb0b4358bb3658ca5c98d25be8892ab.pdf)
Defines latent tokens within the **column space of the vocabulary matrix** $W_\text{unembed}$. Frames latent reasoning as vocabulary superposition: each step is a weighted combination over the vocabulary embedding space rather than a one-hot token. On high-difficulty datasets outperforms COCONUT (last-hidden-state methods); on low-difficulty datasets matches CoT-SFT while significantly shortening chains. **Framework instantiation**: $\text{core}_S$ lives strictly between $S_\mathcal{N}$ (one-hot argmax) and $S_\mathcal{I}$ (arbitrary hidden state) on the vocab-grounding axis — specifically in the column space of $W_\text{unembed}$. A finer-grained position than Soft Thinking on the same axis.

**Tack et al. 2025/2026 — LLM Pretraining with Continuous Concepts (CoCoMix)**
*arXiv:* [2502.08524](https://arxiv.org/abs/2502.08524) · ICLR 2026
Pretraining framework that **predicts continuous concepts extracted from a pretrained sparse autoencoder (SAE)** and interleaves their compressed continuous forms with token hidden representations. The "vocabulary" here is the SAE's concept dictionary — interpretable, sparse, and mechanistically grounded in the model's own activations. Outperforms standard next-token prediction, knowledge distillation, and pause tokens; more sample efficient. **Framework instantiation**: a novel $\text{core}_S$ design where the standardisation vocabulary is **SAE-derived rather than the raw token vocabulary** — interpretable by construction, unlike the arbitrary hidden state in $S_\mathcal{I}$.

**Wang et al. 2023/2024 — Guiding Language Model Reasoning with Planning Tokens**
*arXiv:* [2310.05707](https://arxiv.org/abs/2310.05707) · COLM 2024
Inserts a **single learnable planning token at the start of each reasoning step**, trained as part of the model with 0.001% parameter increase. Addresses the finding that LLMs manage individual reasoning steps well but struggle with consistency across a full chain. **Framework instantiation**: the **thinnest possible trainable standardisation on top of CoT** — $\text{core}_S$ is a single learned discrete-but-trained symbol per step, $k_\text{in} = k_\text{out} = 0$. Anchors the low-capacity end of the core_S design space.

---

## 4. Related Latent-Reasoning Methods (To Be Mapped Into the Framework)

These need their $(k_\text{in}, k_\text{out}, S, \text{training objective})$ tuples spelled out for the design-space figure.

- **Shen et al. 2025 — CODI: Compressing Chain-of-Thought into Continuous Space via Self-Distillation** — [arXiv:2502.21074](https://arxiv.org/abs/2502.21074)
- **HybridCoT (Shen et al. 2025, NeurIPS workshop)** — Interleaves latent and text CoT segments. [OpenReview](https://openreview.net/forum?id=NRGRrHmq1H)
- **Zhu et al. 2025 — Reasoning by Superposition: A Theoretical Perspective on Chain of Continuous Thought** — [arXiv:2505.12514](https://arxiv.org/abs/2505.12514) — Theoretical explanation for why continuous-thought reasoning supports BFS-like exploration. Companion to Deng 2025 (§3) at the theory end.

---

## 5. Mechanistic Interpretability Foundations

Concepts that underpin the geometric and read-out arguments used throughout §1 and §3.

**nostalgebraist 2020 — Interpreting GPT: The Logit Lens** *(blog post, no arXiv)*
[lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens)
The original informal proposal to apply the final unembedding matrix $W_U$ directly to intermediate hidden states $h^{(l)}$ to read out layer-by-layer token predictions. Defined logit-space and the concept of **alignment-with-output-distribution as a function of depth**. Foundational for the tuned lens (Belrose et al. 2023, §1b) and for Wendler et al.'s multilingual analysis (§1a).

**Elhage et al. 2022 — Toy Models of Superposition**
Anthropic: [transformer-circuits.pub/2022/toy_model/index.html](https://transformer-circuits.pub/2022/toy_model/index.html)
Demonstrates that a model with $d$ dimensions can represent **more than $d$ features simultaneously** when features are sparse, by encoding them as nearly-orthogonal directions accepting small cross-feature interference. Establishes the geometric signature of superposition: features arranged in antipodal pairs, pentagons, and other polytopes that pack near-orthogonal vectors into $d$ dimensions. **Foundational** for understanding why hidden states at the body boundary may be in superposition (Deng 2025, Lei 2025) and why SAE-based approaches (CoCoMix) are necessary to disentangle them.

---

## 6. Surveys and Position Pieces

- **Implicit Reasoning in Large Language Models: A Comprehensive Survey** — [arXiv:2509.02350](https://arxiv.org/abs/2509.02350) — Tabulates COCONUT, Soft Thinking, CODI, and similar methods. Useful for the related-work section.

---

## Notes for Stage 1 Drafting

- The Layer Specialisation section (§1) is the empirical backbone of the framework's motivation. The five sub-categories (1a–1e) are independent lines of evidence that converge on the same partition, which is rhetorically the strongest possible framing.
- Multilingual narrative: Wendler, Wu, Zhao, Dumas (causal), and Fierro (factual recall) together. Dumas upgrades the story from correlational to interventional. Fierro adds a clean functional decomposition (subject-enrichment / object-extraction) that mirrors body / N_out.
- Stage-of-inference narrative: Lad gives the four-stage hypothesis directly; Belrose + Geva give the mechanistic vocabulary-projection story; nostalgebraist (§5) is the historical anchor for logit-space readouts.
- Redundancy / fragility narrative: Gromov + Skean give the pruning and probing evidence.
- Convergence narrative (§1e): the Platonic Representation Hypothesis (Huh) is the umbrella that makes the body's representational format model-agnostic by construction, not by accident.
- **RYS series (§2) is the strongest independent confirmation** the framework has. The ~15-layer boundary constant and the leaderboard-anchored "duplication improves, modification breaks" signature are unusually clean operational tests. Use these to frame the framework's choice of $k_\text{in}, k_\text{out}$ as empirically motivated rather than hyperparameter folklore.
- Standardisation function design space (§3): on the vocab-grounding axis, the new entries (Lei = vocab-column-space superposition; Tack/CoCoMix = SAE-dictionary superposition; Wang = single-token planning symbol) populate the spectrum between $S_\mathcal{N}$ and $S_\mathcal{I}$ at finer granularity than before. Deng (chain-of-superposition theory) gives the underlying argument for why these all work: each $\text{core}_S$ pass refines a superposition rather than collapsing to a single hypothesis.
- Mechanistic foundations (§5): Elhage's superposition geometry is the *why* behind Deng/Lei's chain-of-superposition framing; without it, "superposition" in the reasoning-loop sense reads as metaphor rather than mechanism.
- Open citation work: locate Schut et al. (multilingual probing, similar findings to Wendler) and Fan et al. 2019 (LayerDrop) to strengthen §1c. Also consider Huginn (recurrent-depth) for §3 if the looped-architecture axis becomes prominent.
- The empirical caveats — exact values of $k_\text{in}, k_\text{out}$, gradual rather than sharp transitions, decoder-only scope, **within-body structure surfaced by RYS sub-block duplication** — should be cited honestly in the limitations section, drawing on Skean's per-layer probing curves and the RYS series' depth-variability data.
