# The Tensor Is The Message

*A metatheoretical & code-forward whitepaper on N-D inputs, information bottlenecks, and rate–distortion optimal LLMs.*

> “A one-dimensional prompt is a shadow on the wall; the task is cast in many dimensions.”  
> “Don’t make the model **guess** the geometry of meaning—show it.”  
> “Maximize task-relevant mutual information per token.”

---

## Abstract

We argue—and show in code—that **multidimensional (N-D), synchronized inputs** are information-theoretically required for minimum-distortion prediction under a fixed token/compute budget. Through the twin lenses of the **Information Bottleneck (IB)** and **Rate–Distortion (R–D)** frameworks we obtain two key results:

* If any extra field (layout, time, depth, gaze, geo, provenance, etc.) adds **conditional mutual information** about the task beyond text, then **no text-only system can match the Bayes error** of an N-D system at any rate (DPI + Fano).
* At a fixed token budget \(K\), the **N-D feasible set strictly contains** the text-only set; hence \(D^\star_{\text{N-D}}(K) \le D^\star_{\text{text}}(K)\) with strict improvement when the new field is task-informative (R–D dominance).

We translate these proofs into **code scaffolding** for ND-LLM:

* a variable-rate **Token Bottleneck** that allocates \(K\) tokens to the most informative, co-registered cells;
* a pluggable **Field Registry** for synchronized N-D inputs;
* **InfoNCE mutual-information proxies** to maximize \(I(Y;Z)\) per token;
* an evaluation suite that plots empirical R–D curves and Fano-consistent error bounds (see `scripts/rd_audit.py`);
* a memory integration point (**Semantic Tensor Memory**) with holographic superpositions plus constraint modules, and an **Auto-IB Orchestrator** sketch for adaptive data/route selection.

> “Structure that is known to the data should be paid once in the **input**, not many times in **inference**.”

---

## 1. Metatheory in Three Boxes

### Box A — Necessity (IB + DPI)

Let \(W\) be world state, inputs \(X_{1:M}\) be synchronized fields (text, layout, time, depth, …), task \(Y\), encoder \(Z = \phi(X_{1:M})\).

If \(I\!\left(Y;X_{1:M}\right) > I\!\left(Y;X_{\text{text}}\right)\), then for any text-only encoder \(\phi_{\text{text}}\):

\[
\max_{\phi_{\text{text}}} I\!\left(Y;\phi_{\text{text}}(X_{\text{text}})\right) \le I\!\left(Y;X_{\text{text}}\right) < I\!\left(Y;X_{1:M}\right) \le \max_{\phi} I\!\left(Y;\phi(X_{1:M})\right).
\]

Therefore text-only is strictly sub-optimal when extra fields carry task-relevant information. By **Fano**, a utility gap \(\Delta I\) yields a strictly worse lower bound on error.

### Box B — Dominance at Fixed Rate (R–D)

Define

\[
R^\star(D) = \min_{p(z\mid x)} I(X;Z) \quad \text{s.t.}\quad \min_f \mathbb{E}[d(Y, f(Z))] \le D.
\]

Since text-only encoders are a subset of N-D encoders,

\[
R^\star_{\text{N-D}}(D) \le R^\star_{\text{text}}(D) \quad \Longleftrightarrow \quad D^\star_{\text{N-D}}(K) \le D^\star_{\text{text}}(K),
\]

with strict improvement when the added field is task-informative.

### Box C — Synchronization

Let \(\phi_m : \Omega_m \to \Omega^*\) map each field’s native domain to a canonical space (e.g., page UV / token spans). Build cell-wise tokens

\[
Z_j = f\!\Big(E_{\text{vis}}(C_j), \{\mathrm{pool}_{\xi\in\phi_m^{-1}(C_j)} g_m(X_m(\xi))\}_m\Big).
\]

**Registration error** lowers \(I(Y;\tilde X_{1:M})\); canonicalize first (layout coords, UV unwarp, timestamps, geo normalization).

> “If it aligns, it informs. If it’s misaligned, it confuses.”

---

## 2. Design Laws (Quotable)

1. **Law of Paid Geometry.** Put spatial/temporal/causal structure in the *input*, not the *inference*.
2. **Mutual Information per Token.** Optimize \(I(Y;Z)/\lvert Z \rvert\). Rate is a budget; spend it where it buys \(\Delta I\).
3. **Dominance of N-D.** If \(I(Y;X_{\neg \text{text}} \mid X_{\text{text}}) > 0\), N-D strictly dominates at equal rate.
4. **Variable-Rate or Bust.** One compression level cannot be optimal across queries; learn to adapt \(K\).
5. **Explainable Bottlenecks.** A visible \(Z\) is a natural explanation surface (what was kept vs. dropped).

---

## 3. Minimal, Composable Implementation (PyTorch-flavored pseudocode)

The following snippets mirror the intent of the `nd_llm` codebase while remaining framework-agnostic enough to port.

### 3.1 Field Registry (N-D plug-ins)

```python
# src/nd/field_registry.py
from dataclasses import dataclass
from typing import Callable, Optional, Dict, Any
import torch
import torch.nn as nn

@dataclass
class FieldSpec:
    name: str
    encoder: nn.Module               # raw -> tokens (B, N_m, d_m)
    align: Callable[[Any], torch.Tensor]  # maps native coords -> canonical Ω*: returns (B, N_m, 2) or (B, N_m, k)
    reducer: str = "mean"            # how to pool within a cell
    proj: Optional[nn.Module] = None # to project d_m -> d_model

class FieldRegistry(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.fields: Dict[str, FieldSpec] = nn.ModuleDict()

    def register(self, spec: FieldSpec):
        if spec.proj is None:
            spec.proj = nn.Linear(spec.encoder.out_dim, self.d_model)
        self.fields[spec.name] = spec

    def forward(self, batch: Dict[str, Any]):
        """
        Returns list of dicts: [
            {
                "name": str,
                "tokens": Tensor(B, N, d_model),
                "coords": Tensor(B, N, k) in Ω*,
            },
            ...,
        ]
        """
        outputs = []
        for name, spec in self.fields.items():
            raw = batch[name]                    # raw payload (images, text ids, etc.)
            tok = spec.encoder(raw)              # (B, N_m, d_m)
            tok = spec.proj(tok)                 # (B, N_m, d_model)
            coords = spec.align(batch)           # (B, N_m, k) in Ω*
            outputs.append({"name": name, "tokens": tok, "coords": coords})
        return outputs
```

> “Every field gets a seat at the table **iff** it can sit in the same coordinate system.”

### 3.2 Canonical Cell Builder & Aggregation

```python
# src/nd/canonical_cells.py
import torch
from typing import List, Dict

def rasterize_cells(batch_size: int, grid_hw=(32, 24), device="cuda"):
    H, W = grid_hw
    ys = torch.linspace(0, 1, H, device=device)
    xs = torch.linspace(0, 1, W, device=device)
    Y, X = torch.meshgrid(ys, xs, indexing="ij")
    centers = torch.stack([Y, X], dim=-1).view(1, H * W, 2).repeat(batch_size, 1, 1)
    return centers

def assign_to_cells(coords: torch.Tensor, cell_centers: torch.Tensor, tau=0.1):
    diff = coords.unsqueeze(2) - cell_centers.unsqueeze(1)
    dist2 = (diff ** 2).sum(-1)
    weights = torch.softmax(-dist2 / (2 * tau * tau), dim=-1)
    return weights

def aggregate_fields(field_tokens: List[Dict], cell_centers, agg="mean"):
    B, C, _ = cell_centers.shape
    d_model = field_tokens[0]["tokens"].shape[-1]
    fused = torch.zeros(B, C, d_model, device=cell_centers.device)
    for ft in field_tokens:
        tok, coords = ft["tokens"], ft["coords"]
        weights = assign_to_cells(coords, cell_centers)
        fused += torch.einsum("bnc,bnd->bcd", weights, tok)
    return fused
```

### 3.3 Variable-Rate Token Bottleneck (Top-K with Learned Scores)

```python
# src/nd/token_bottleneck.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class TokenBottleneck(nn.Module):
    def __init__(self, d_model: int, scorer_hidden=256):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, scorer_hidden),
            nn.ReLU(),
            nn.Linear(scorer_hidden, 1),
        )

    def forward(self, z_cells: torch.Tensor, K: int, soft=False, tau=0.5):
        B, C, D = z_cells.shape
        scores = self.scorer(z_cells).squeeze(-1)
        if soft:
            gumbel = -torch.log(-torch.log(torch.rand_like(scores)))
            topk = (scores + gumbel) / tau
            weights = F.softmax(topk, dim=-1)
            z_soft = torch.einsum("bc,bcd->bd", weights, z_cells)
            return z_soft.unsqueeze(1).repeat(1, K, 1), None
        topk = torch.topk(scores, k=K, dim=-1)
        idx = topk.indices
        gathered = torch.gather(z_cells, 1, idx.unsqueeze(-1).repeat(1, 1, D))
        return gathered, idx
```

> “Variable rate is not a feature; it is the *optimum*.”

### 3.4 Mutual-Information Proxy (InfoNCE)

```python
# src/nd/mi_proxy.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class MIProxy(nn.Module):
    def __init__(self, d_model: int, d_proj: int = 256, temperature: float = 0.07):
        super().__init__()
        self.f = nn.Sequential(
            nn.Linear(d_model, d_proj),
            nn.ReLU(),
            nn.Linear(d_proj, d_proj),
        )
        self.h = nn.Sequential(
            nn.Linear(d_model, d_proj),
            nn.ReLU(),
            nn.Linear(d_proj, d_proj),
        )
        self.tau = temperature

    def forward(self, z: torch.Tensor, y_repr: torch.Tensor):
        pooled = z.mean(dim=1)
        pooled = F.normalize(self.f(pooled), dim=-1)
        target = F.normalize(self.h(y_repr), dim=-1)
        logits = (pooled @ target.T) / self.tau
        labels = torch.arange(pooled.size(0), device=pooled.device)
        loss = F.cross_entropy(logits, labels)
        return -loss, logits
```

### 3.5 End-to-End Skeleton

```python
# src/nd/model.py
import torch
import torch.nn as nn

from .field_registry import FieldRegistry
from .canonical_cells import rasterize_cells, aggregate_fields
from .token_bottleneck import TokenBottleneck
from .mi_proxy import MIProxy

class NDEncoderDecoder(nn.Module):
    def __init__(self, d_model=768, grid_hw=(32, 24), decoder: nn.Module | None = None):
        super().__init__()
        self.registry = FieldRegistry(d_model=d_model)
        self.grid_hw = grid_hw
        self.bottleneck = TokenBottleneck(d_model)
        self.decoder = decoder
        self.mi = MIProxy(d_model)

    def register_field(self, spec):
        self.registry.register(spec)

    def forward(self, batch, K_tokens: int):
        field_tokens = self.registry(batch)
        centers = rasterize_cells(
            batch_size=field_tokens[0]["tokens"].size(0),
            grid_hw=self.grid_hw,
            device=field_tokens[0]["tokens"].device,
        )
        z_cells = aggregate_fields(field_tokens, centers)
        z_sel, idx = self.bottleneck(z_cells, K=K_tokens)
        logits = self.decoder(context=z_sel, targets=batch["targets"])
        mi_lb, _ = self.mi(z_sel, batch["target_repr"])
        return logits, {"mi_lb": float(mi_lb.detach().cpu()), "idx": idx}
```

### 3.6 Training Loop + Loss

```python
# src/train.py
import torch
import torch.nn.functional as F

def loss_fn(logits, targets, mi_lb, alpha_rate=1e-4, beta_mi=0.1, K_used=None):
    ce = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        targets.view(-1),
        ignore_index=-100,
    )
    rate_pen = alpha_rate * (K_used if K_used is not None else 0.0)
    mi_pen = -beta_mi * mi_lb
    return ce + rate_pen + mi_pen, {"ce": float(ce), "rate": float(rate_pen), "mi": float(mi_pen)}

def train_step(model, batch, K, opt):
    logits, logs = model(batch, K_tokens=K)
    loss, parts = loss_fn(logits, batch["targets"], logs["mi_lb"], K_used=K)
    loss.backward()
    opt.step()
    opt.zero_grad()
    parts.update(logs)
    return {k: float(v) for k, v in parts.items()}
```

### 3.7 Token-Equalized Evaluation (R–D sweep)

```python
# src/eval_rd.py
import json
import os
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F

@torch.no_grad()
def eval_rd(model, dataloader: Iterable, Ks=(64, 128, 256, 400)):
    results = []
    for K in Ks:
        metrics = {"K": K, "ce": [], "mi_lb": []}
        for batch in dataloader:
            logits, logs = model(batch, K_tokens=K)
            ce = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                batch["targets"].view(-1),
                ignore_index=-100,
            )
            metrics["ce"].append(float(ce))
            metrics["mi_lb"].append(float(logs["mi_lb"]))
        results.append({k: (np.mean(v) if isinstance(v, list) else v) for k, v in metrics.items()})
    os.makedirs("experiments", exist_ok=True)
    with open("experiments/rd_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
```

> “Plot \(K\) vs. distortion. The curve doesn’t lie.”

---

## 4. What to Measure (and Why It Matters)

* **Primary R–D:** map \(K \rightarrow\) task distortion (CER/WER/EM/F1). Expect N-D curves **below** text-only when fields are informative.
* **\(\widehat I(Y;Z)\) proxy:** InfoNCE lower bound; expect negative correlation with distortion across \(K\).
* **Fano sanity:** track \(\widehat H(Y \mid Z) \approx \text{CE}\) and show lower error bounds shrink as \(\widehat I\) rises.
* **Registration delta:** measure utility before/after canonicalization—UV warp, layout coords, time normalization.
* **Budget neutrality:** keep decoder token budget fixed; report **FLOPs/latency** separately for front-end encoders.

---

## 5. Taxonomy of Synchronizable Fields (and their \(\Delta I\))

| Field type | Marginal utility | Example tasks |
| --- | --- | --- |
| 2-D layout | Column order, reading flow, key–value linking | Document QA, key information extraction |
| Time | Temporal ordering, causal signals | Video QA, ASR alignment, dialogue |
| Depth/geometry | Curvature, occlusion, 3-D relation | Curved scans, robotics grounding |
| Gaze/salience | Focus-of-attention priors | UX analytics, salience-weighted QA |
| Geo | Where/when grounding | Timelines, travel itineraries |
| Provenance | Source credibility, citation control | Grounded answering, refusal triggers |
| Interaction | Click/scroll/keystroke traces | UI understanding, Pix2Struct-style tasks |

> “If a field answers a question humans ask, it raises \(I(Y;X)\).”

---

## 6. Auto-IB Orchestrator (Sketch)

1. **Predict bottlenecks:** meta-model forecasts \(\widehat{\Delta I}\) for candidate fields; chooses \((\text{fields}, K)\) per query.
2. **Proxy trials:** cheap sub-runs validate predicted gains (guardrails vs. Goodhart).
3. **Hard-case synthesis:** generate adversarial distractors/missing-evidence samples; fine-tune bottleneck.
4. **Telemetry → policy:** adjust token router priors, field thresholds, chunking schedules.

---

## 7. Semantic Tensor Memory (STM) Hooks

* **Write:** capture `{pipeline, fields, K, metrics, mi_lb, idx_cells, artifacts(layout, uv, time)}` per evaluation.
* **Read:** retrieve nearest memories by `{task, layout_sig, curvature_idx, time pattern}`; inject as few-shot/latent hints **without raising live \(K\)**.
* **Explain:** show kept/dropped cells & field attributions; trend \(\widehat I\) vs. error over time.

---

## 8. Evaluation Matrix (Token-Equalized)

| Domain | Baseline (Text) | N-D Inputs | Metrics |
| --- | --- | --- | --- |
| Long QA (wiki) | truncate/summary | salience + provenance | EM/F1 vs \(K\) |
| DocVQA/forms | OCR concat | 2-D layout (+ image) | EM/F1, TEDS vs \(K\) |
| Tables | linearized CSV | 2-D grid coords | exact cell EM vs \(K\) |
| Video QA | sampled frames text | time + frames | EM/F1 vs \(K\) |
| Curved scans | OCR text | depth/UV | CER/WER vs \(K\) |
| Geo-timeline | text only | geo + time | EM/F1 vs \(K\) |

Ablate each field: remove one at a time, noise it, or randomize registration. R–D curves should regress toward text-only.

---

## 9. Safety, Privacy, Bias

* **Sensitive fields** (geo, biometrics): minimize, aggregate, or encrypt; add consent gates and redaction policies.
* **Routing bias audits:** ensure token allocation isn’t demographically skewed; load-balance MoE routes.
* **Grounded abstention:** explicit *not-enough-info* heads for out-of-scope; penalize unsupported generations.
* **Provenance hygiene:** prevent source embeddings from collapsing into protected-attribute proxies; run counterfactual checks.

> “Better grounding lowers hallucination; careless fields raise new risks. Measure both.”

---

## 10. Limitations

* \(\widehat I\) (InfoNCE) is a **proxy**, not absolute MI. Use **comparatively**.
* Registration quality is a hard limiter; garbage in, garbage out.
* Encoder cost vs. decoder savings: report both; pick Pareto-efficient points.
* N-D helps where fields are **relevant**; pure lexical tasks may see parity.

---

## 11. Reproducibility Checklist

1. Fix the decoder family; change only inputs/routing.
2. Token budgets \(K \in \{64, 128, 256, 400\}\), three seeds; report mean ± SEM.
3. Log `{K, distortion, Ĩ, FLOPs, latency}` per run.
4. Generate figures from JSON logs; commit configs & hashes.

---

## 12. Frequently Quoted Sentences

* “**N-D or Nothing**: if the world offers task-relevant fields beyond text, a text-only model is provably sub-optimal.”
* “**Spend tokens where they buy mutual information**—not where they soothe our linear habits.”
* “**Registration is destiny**: alignment turns extra modalities into extra evidence.”
* “The bottleneck \(Z\) is where your explanations live: *what survived is what mattered*.”
* “Hallucination is often **projection debt**; N-D inputs pay it down at the source.”

---

## 13. Appendix A — Minimal YAML Configs

```yaml
# configs/nd_text_layout.yaml
model:
  d_model: 768
  grid_hw: [32, 24]
  decoder: "tiny_seq2seq"
fields:
  - name: "text"
    encoder: "bert_tiny"
    align: "spans_to_uv"
  - name: "layout"
    encoder: "bbox_encoder"
    align: "bbox_to_uv"
train:
  lr: 3e-4
  steps: 20000
  batch_size: 8
sweep:
  K: [64, 128, 256, 400]
log:
  out_dir: "experiments"
```

---

## 14. Appendix B — Axioms → Algorithms Cheat-Sheet

* **DPI (+) chain rule → Text deficit test.** Estimate \(\widehat{\Delta I} = \widehat I(Y;X_{1:M}) - \widehat I(Y;X_{\text{text}})\). If positive, route that field.
* **R–D set inclusion → Token-equalized curves.** Compare distortions for each \(K\); N-D should be \(\le\) text.
* **Fano bound → Refusal training.** Learn “not enough info” when \(H(Y \mid Z)\) is high.
* **IB Lagrangian \(\min H(Y \mid Z) + \beta I(X;Z)\) → Loss.** Use CE + \(\alpha \lvert Z \rvert\) − \(\beta \widehat I\).

---

## 15. Closing

> “Language is a projection; the world is a tensor. If you want minimum-distortion predictions at fixed budget, **model the tensor** and **bottleneck the projection**.”

This whitepaper gives the math, the design laws, and the code scaffolding to do exactly that. Plug in fields, canonicalize, route \(K\) tokens by mutual information, and let the R–D curves tell the story.

---

### Quick Start (developer capsule)

1. Implement two fields (text + layout) via `FieldRegistry`; feed any doc dataset with boxes.
2. Train `NDEncoderDecoder` with \(K \in \{64, 128, 256, 400\}\); log CE and \(\widehat I\).
3. Plot \(K\) vs distortion vs baseline (text-only).
4. Add time/depth/geo as needed; repeat.
5. Wire STM logging keys `{fields, K, mi_lb, kept_cells}` for explainability & retrieval.

> “Plot first. Argue later.”
