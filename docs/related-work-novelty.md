# Related Work & Novelty (as of Oct 22, 2025)

## Arbitrary-input & Generalist Backbones

**Perceiver IO** maps *arbitrary input arrays → arbitrary output arrays* through a fixed-size **latent bottleneck**, offering a domain-agnostic fusion core. **Gato** serializes many modalities and actions into one transformer. **Flamingo** bridges vision encoders to language models via a **Perceiver-style resampler** to pass few, strong visual tokens. **Kosmos-1** aligns perception with language for in-context, multimodal instruction following. [1]

## Information Bottleneck (IB) & Rate–Distortion (R–D) in (M)LLMs

Information Bottleneck has been extended to multimodal fusion (MIB/OMIB) and to compression via VIB-guided pruning and merging; recent work formalizes **prompt compression** and multi-agent **R–D-optimized communication**. These typically optimize *modules* (tokens, layers, channels) or *links* (agent messaging), rather than governing the entire model–memory–reasoning stack. [2]

## External & Non-Parametric Memories

**REALM** pretrains with a learned retriever; **RETRO** trains and infers with massive external text memories; **Memorizing Transformers** add approximate kNN memory at inference; **kNN-LM** (and efficient variants) interpolate parametric language models with datastore lookups. These are powerful, but they are **document/key–value memories**, not holographic tensor stores with write/read *policies*. [3]

## Tensor-Native / Holographic Representations & Neuro-Symbolic Layers

Classical **Tensor Product Representations (TPR)**, **Holographic Reduced Representations (HRR)**, and **Hyperdimensional Computing** show superposition and binding for symbolic structure in vectors and tensors. **Logic Tensor Networks** and **TensorLog** make logic differentiable over tensor and factor-graph computations—adjacent to the proposed **tensor-native memory plus constraints** layer. [4]

## Layout & Field-Structured Grounding (Beyond “Just Vision + Text”)

**LayoutLM** (text + layout + image) and **Donut** (OCR-free document understanding) show that **adding non-linguistic fields** (layout, structure) materially reduces distortion—evidence for treating *fields* as first-class entities. [5]

---

## Novelty in This Work (Defensible Claim)

1. **Fields-as-first-class, synchronized inputs** (not only modalities): layout, time, tool or program state, and sensor bands are aligned and reasoned over together. Prior systems accept many inputs but do not **operationalize “field synchrony”** as a core abstraction. [1]
2. **A single, global IB/R–D controller** that jointly allocates the *rate budget* across **(a)** field selection, **(b)** tokenization density, **(c)** memory writes and reads, and **(d)** reasoning depth—optimizing task distortion under a unified objective. Existing IB/R–D works address pieces (token pruning, prompt compression, inter-agent communication), not the **end-to-end policy**. [6]
3. **Semantic Tensor Memory (STM)** as a persistent, **holographic** store with content-addressable reads and superposed bindings, with **IB-governed write/read policies**—distinct from document retrieval or flat kNN caches. [4]
4. **Neuro-symbolic, tensor-native constraints** (LTN/TensorLog-style) integrated *inside* the inference loop so logical constraints propagate bidirectionally across fields and memory, not merely as post-hoc checks. [7]

> **Novelty statement.** *No prior work simultaneously (i) formalizes multi-**field** grounding and synchrony, (ii) **optimizes one IB/R–D objective** across fields → tokens → memory → steps, and (iii) integrates a **tensor-native holographic memory** with a neuro-symbolic constraint layer for **end-to-end** reasoning.* (See closest neighbors above.)

---

## Minimal Schematic (Architecture at a Glance)

```
[Field adapters: text | vision | layout | time | tool-state | sensors]
             └──> Field-sync aligner  ───────┐
                                             │
                             [Global IB/R–D Controller]
           allocates rate across: fields ▸ tokens ▸ memory ▸ reasoning
                                             │
                 ┌───────────── Core LLM Reasoner ─────────────┐
                 │                                             │
           [STM: Semantic Tensor Memory]  ⇄  [Neuro-Symbolic Constraints]
       (holographic writes/reads; IB policies)   (LTN/TensorLog-style)
                 └──────────── evidence-bound outputs ─────────┘
```

## Suggested Ablations to Validate Originality

* **Disable** the global IB controller → expect higher distortion at the same token or step budget.
* **Swap STM with kNN/RAG** → expect worse long-horizon consistency under an equal rate budget.
* **Remove constraints** → expect more contradiction or violation at equal rate.

---

## References

[1]: https://arxiv.org/abs/2107.14795 "Perceiver IO: A General Architecture for Structured Inputs & Outputs"
[2]: https://arxiv.org/abs/2210.17444 "Multimodal Information Bottleneck: Learning Minimal Sufficient Unimodal and Multimodal Representations"
[3]: https://proceedings.mlr.press/v119/guu20a/guu20a.pdf "REALM: Retrieval-Augmented Language Model Pre-Training"
[4]: https://www.lscp.net/persons/dupoux/teaching/AT1_2014/papers/Smolensky_1990_TensorProductVariableBinding.AI.pdf "Tensor Product Variable Binding and the Representation of Symbolic Structure"
[5]: https://arxiv.org/pdf/1912.13318 "LayoutLM: Pre-training of Text and Layout for Document Understanding"
[6]: https://arxiv.org/abs/2407.15504 "Fundamental Limits of Prompt Compression"
[7]: https://www.ijcai.org/proceedings/2017/0221.pdf "Logic Tensor Networks for Semantic Image Interpretation"
