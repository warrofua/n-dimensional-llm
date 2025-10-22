# Why N-D (tensor) inputs are necessary for minimum-distortion prediction

Below is a compact, first-principles derivation with the Information Bottleneck (IB) and Rate–Distortion (R–D) lenses. It formalizes your intuition: if reality is multi-field, then projecting to 1-D text imposes an irreducible information deficit for many tasks; synchronized N-D inputs remove that deficit and dominate in R–D.

## 1) Setup

Let the world state be a random object 𝑊.

We observe 𝑀 synchronized fields 𝑋₁:𝑀 = (𝑋₁,…,𝑋𝑀) derived from 𝑊: e.g., text, 2-D layout, depth, time, gaze, geo, provenance.

A task variable 𝑌 (answer, label, program output) depends on 𝑊.

An encoder produces a representation 𝑍 = 𝜙(𝑋₁:𝑀) that the decoder uses to predict 𝑌̂.

**Text-only projection.** Many pipelines use a lossy projection 𝑋ₜₑₓₜ = 𝑇(𝑊) (OCR/string), then 𝑍 = 𝜙ₜₑₓₜ(𝑋ₜₑₓₜ).

We care about two fundamental quantities:

* **Utility:** 𝐼(𝑌;𝑍) — task-relevant information that survives the bottleneck.
* **Rate:** 𝐼(𝑋₁:𝑀;𝑍) or a proxy (e.g., number of tokens 𝐾) — how much we spend to carry info forward.

## 2) Necessity via mutual information (IB view)

**Proposition 1 (Information deficit of 1-D projection)**

If any field adds task-relevant info beyond text, i.e.

𝐼(𝑌;𝑋₁:𝑀) > 𝐼(𝑌;𝑋ₜₑₓₜ),

then no text-only encoder can match the best N-D encoder’s utility:

max₍𝜙ₜₑₓₜ₎ 𝐼(𝑌;𝜙ₜₑₓₜ(𝑋ₜₑₓₜ)) ≤ 𝐼(𝑌;𝑋ₜₑₓₜ) < 𝐼(𝑌;𝑋₁:𝑀) ≥ max₍𝜙₎ 𝐼(𝑌;𝜙(𝑋₁:𝑀)).

*Proof (sketch).* Data processing inequality (DPI): for any mapping 𝜙,

𝐼(𝑌;𝜙(𝑋)) ≤ 𝐼(𝑌;𝑋).

Chain rule gives

𝐼(𝑌;𝑋₁:𝑀) = 𝐼(𝑌;𝑋ₜₑₓₜ) + 𝐼(𝑌;𝑋¬ₜₑₓₜ ∣ 𝑋ₜₑₓₜ).

If the conditional term > 0, text-only lacks that increment forever. ∎

**Error bound consequence (Fano).** For discrete 𝑌,

𝑃ₑ ≥ (𝐻(𝑌∣𝑍) − 1)/log∣𝑌∣ ≥ (𝐻(𝑌) − 𝐼(𝑌;𝑍) − 1)/log∣𝑌∣.

Thus a utility gap Δ𝐼 = 𝐼(𝑌;𝑋₁:𝑀) − 𝐼(𝑌;𝑋ₜₑₓₜ) > 0 implies a strictly worse lower bound on error for all text-only systems.

*Interpretation.* If layout, time, depth, gaze, geo, provenance, etc. carry any conditionally new information about 𝑌 beyond text, then a 1-D pipeline is provably sub-optimal: it cannot reach the Bayes error of an N-D pipeline.

## 3) Dominance at fixed rate (R–D view)

Define the prediction R–D problem with a task distortion 𝑑(𝑌,𝑌̂):

𝑅⋆(𝐷) = min₍𝑝(𝑧∣𝑥)₎ 𝐼(𝑋;𝑍)  s.t.  min₍𝑓₎ 𝐸[𝑑(𝑌,𝑓(𝑍))] ≤ 𝐷.

Compare two feasible sets:

* **Text-only:** 𝑋 = 𝑋ₜₑₓₜ.
* **N-D tensor:** 𝑋 = 𝑋₁:𝑀.

**Proposition 2 (Rate advantage with extra fields)**

For any target distortion 𝐷,

𝑅⋆_{N-D}(𝐷) ≤ 𝑅⋆_{text}(𝐷),

with strict inequality whenever the added fields are task-informative:

𝐼(𝑌;𝑋₁:𝑀) > 𝐼(𝑌;𝑋ₜₑₓₜ).

*Proof (sketch).* Any text-only encoder 𝑝(𝑧∣𝑥ₜₑₓₜ) is a special case of an N-D encoder that ignores extra fields, so the N-D feasible set is a superset in the R–D program. Strict inclusion + task relevance yields strict improvement. ∎

Equivalently, at a fixed rate (e.g., token budget 𝐾),

𝐷⋆_{N-D}(𝐾) ≤ 𝐷⋆_{text}(𝐾),

with a strict gap under the same condition.

*Interpretation.* With the same number of tokens/compute, the N-D encoder achieves equal or lower distortion; when new fields matter, it is strictly better.

## 4) Why synchronization (tensorization) matters

Extra fields must be co-registered to be useful. Let 𝜙ₘ: Ωₘ → Ω* map each field’s native domain (pixels, time, geo) into a canonical space (e.g., page UV, token spans). Then build tokens

𝑍ⱼ = 𝑓 (𝐸ᵥᵢₛ(𝐶ⱼ), {pool_{ξ∈𝜙ₘ⁻¹(𝐶ⱼ)} 𝑔ₘ(𝑋ₘ(ξ)) }ₘ ),

over cells 𝐶ⱼ ⊂ Ω*.

If 𝜙ₘ is inaccurate, you pay a registration penalty:

𝐼(𝑌;𝑋₁:𝑀) − 𝐼(𝑌;𝑋̃₁:𝑀) = loss from misalignment.

Hence the practical recipe: canonicalize first (e.g., layout coordinates, UV unwarp, time normalization), then compress.

## 5) When does text suffice?

Text-only is optimal iff the new fields are conditionally irrelevant:

𝐼(𝑌;𝑋¬ₜₑₓₜ ∣ 𝑋ₜₑₓₜ) = 0.

Examples: pure lexical paraphrase tasks, trivia whose answer is fully in the provided text without spatial/temporal cues. In all other cases (tables, forms, page layout, timelines, diagrams, curved photos, provenance-sensitive QA, salience-sensitive UX), the conditional MI is positive, so N-D is required to reach minimum distortion or minimum error at a fixed rate.

## 6) Practical corollaries (what to build)

* **Variable-rate bottleneck dominates.** Given a fixed token budget 𝐾, learn a selector/router that spends tokens on cells 𝐶ⱼ maximizing an MI proxy with 𝑌. This approximates the IB optimum: maximize 𝐼(𝑌;𝑍) subject to ∣𝑍∣ = 𝐾.
* **Fields shift R–D curves down.** Adding synchronized fields (layout, time, depth, gaze, geo, provenance) increases 𝐼(𝑌;𝑋), so the whole curve 𝐷⋆(𝐾) moves down (or left in 𝑅⋆(𝐷)).
* **Fano sanity check.** Track 𝐼̂(𝑌;𝑍) (e.g., InfoNCE proxy) and verify that decreases in CER/WER or EM/F1 track increases in 𝐼̂. If not, your field isn’t aligned or your selector isn’t using it.
* **Square-root heuristic = intuition, not law.** Your “1-D vs 2-D reduces the ‘unknown manifold’ from
  𝑛
  n to
  𝑛
  n
  ​” is a good mental model of dimensional burden, but the proved quantity is the Δ𝐼 above. Use the heuristic in prose; optimize Δ𝐼 in code.

## 7) Minimal proofs-as-algorithms (plug into your stack)

* **Text deficit test.** Estimate Δ𝐼̂ = 𝐼̂(𝑌;𝑋₁:𝑀) − 𝐼̂(𝑌;𝑋ₜₑₓₜ) with variational bounds (e.g., MINE/InfoNCE) on held-out data. If Δ𝐼̂ > 0, route that field into the bottleneck.
* **R–D dominance check.** Sweep token budgets 𝐾 for text-only vs N-D; plot 𝐾 vs distortion. If curves cross only at degenerate tasks, you have empirical Proposition 2.
* **Registration audit.** Measure 𝐼̂(𝑌;𝑋̃₁:𝑀) before/after canonicalization 𝜙ₘ. Gains here directly predict task gains.
