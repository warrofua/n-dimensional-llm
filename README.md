# N‑Dimensional LLM (ND‑LLM)

**Grounded, multi‑field language models with information‑bottleneck compression and semantic tensor memory.**

> Build LLMs that reason across *N* synchronized fields (text, layout, space, time, sensors, …) while using **variable‑rate token bottlenecks**—guided by information‑bottleneck / rate–distortion objectives—to slash context load without sacrificing fidelity. Persist compressed states in **Semantic Tensor Memory (STM)** and adapt over time via an **Auto‑IB Orchestrator**.

---

## Why N‑D now?

* **Reality is not 1‑D text.** Many tasks require aligned signals (words ↔ layout ↔ time ↔ coordinates ↔ audio/video ↔ metadata).
* **Context is scarce.** Token budgets and latency matter; naive concatenation wastes capacity.
* **Compression should be *learned***—and **query‑aware**—not a post‑hoc heuristic.

---

## Core concepts

### 1) Field Registry & Synchronization

A first‑class schema for declaring input “fields” and their alignments (indices, coordinates, timestamps, salience). The registry is how encoders agree on *where* a token came from and *how* it lines up with others.

* Declarative field types (e.g., `text`, `bbox`, `image_patch`, `timestamp`, `3d_pose`, `audio_chunk`, `sensor_feature`).
* Cross‑field alignment keys (e.g., `doc_id`, `frame_id`, `entity_id`, `t`).
* Optional *salience* / *uncertainty* channels for downstream compression.

### 2) Variable‑Rate Token Bottleneck (IB/RD‑guided)

A learnable compression layer between encoders and the decoder. It chooses **how many** tokens to pass and **which** ones—*per query and per example*—balancing utility vs. cost.

* Optimized with **Information Bottleneck** and **Rate–Distortion** proxies.
* Supports multiple mechanisms: summary tokens, top‑k routing, MoE gates, product quantization, or learned entropy models.
* Exposes a **budget knob** (target bits/tokens) and collects **usage telemetry** for later tuning.

### 3) Semantic Tensor Memory (STM)

A long‑horizon store of compressed states (Z) and metadata. Think: deduped, query‑addressable, versioned *semantic shards*.

* Saves bottleneck outputs + field keys + task labels.
* Enables retrieval of *compressed* context instead of raw history.
* Supports background audits: retention, drift, reconstruction quality.

### 4) Auto‑IB Orchestrator

A control loop that probes what the model remembers, tunes bottleneck budgets, re‑balances per field, and triggers fine‑tuning when drift or loss is detected.

* Active‑testing hooks (e.g., counterfactual prompts, canary questions).
* Budget scheduling (tighten/loosen by task, user, or SLA).
* “Pay with bits only when it pays with accuracy.”

---

## Architecture (high level)

```
        [ Field A ]  →  Encoder_A  ┐
        [ Field B ]  →  Encoder_B  ┼─►  Variable‑Rate Bottleneck  ─►  Decoder/LLM  ─►  Outputs
        [ Field C ]  →  Encoder_C  ┘              │
                                                 ▼
                                            Semantic
                                         Tensor Memory
```

---

## Repo layout (proposed)

```
.
├── nd_llm/                 # Library code (registry, encoders, bottleneck, STM, orchestrator)
│   ├── registry/
│   ├── encoders/
│   ├── bottleneck/
│   ├── stm/
│   ├── orchestration/
│   └── utils/
├── examples/              # Minimal runnable demos / notebooks
├── benchmarks/            # Tasks and evaluation harness
├── scripts/               # Training, evaluation, data prep
├── tests/
├── docs/
│   └── figures/
└── paper/
    └── Toward-N-Dimensional-LLMs-with-Information-Bottlenecks.pdf  # ← add your PDF here
```

---

## Quickstart

> **Status:** research pre‑release. The code skeleton below shows the intended API; swap in your encoders/LLM as you prototype.

### Installation

```bash
# clone
git clone https://github.com/<you>/n-dimensional-llm.git
cd n-dimensional-llm

# (optional) create env
python -m venv .venv && source .venv/bin/activate

# editable install
pip install -e .
```

### “Hello, N‑D” (pseudo‑Python)

```python
from nd_llm import Registry, IBottleneck, STM, Orchestrator
from nd_llm.encoders import TextEncoder, LayoutEncoder
from nd_llm.utils import pack_fields

# 1) Declare fields + alignment
reg = Registry()
reg.add_field("text", keys=["doc_id", "span_id"], salience=True)
reg.add_field("bbox", keys=["doc_id", "span_id"])  # layout boxes per span

# 2) Build encoders
enc_text   = TextEncoder(model="tiny-bert")
enc_layout = LayoutEncoder(kind="xyxy")

# 3) Variable‑rate bottleneck (target ~256 tokens eq.)
ib = IBottleneck(target_budget="256t", objective="ib-rd")

# 4) Semantic Tensor Memory + Orchestrator
stm = STM(store_dir="./stm")
ctl = Orchestrator(stm=stm, bottleneck=ib)

# 5) Ingest a document page
fields = pack_fields(
    text=[{"doc_id":1, "span_id":i, "text":t, "salience":s} for i,(t,s) in enumerate(spans)],
    bbox=[{"doc_id":1, "span_id":i, "xyxy":b} for i,b in enumerate(boxes)],
)

# 6) Encode → compress → decode
Z = ib.compress(
    encs={"text": enc_text(fields["text"]),
          "bbox": enc_layout(fields["bbox"])},
    registry=reg,
    query="Summarize the invoice totals by vendor",
)

answer = your_llm.decode(Z, prompt="Summarize the invoice totals by vendor")
ctl.log_usage(example_id="doc_1", Z=Z, answer=answer)
```

### Minimal field‑registry (YAML)

```yaml
fields:
  text:
    keys: [doc_id, span_id]
    salience: true
  bbox:
    keys: [doc_id, span_id]
  timestamp:
    keys: [doc_id, frame_id, session_id, t]
  audio_chunk:
    keys: [session_id, t]

affinity:
  - [text, bbox, by: [doc_id, span_id]]
  - [text, timestamp, by: [doc_id]]
  - [audio_chunk, timestamp, by: [session_id, t]]
```

---

## Benchmarks (planned)

* **Doc understanding (2‑D → 3‑D time):** forms/receipts across pages and revisions; target equal accuracy with fewer tokens.
* **Video‑text QA (time + layout):** align subtitles, frames, OCR boxes; test variable‑rate compression under latency caps.
* **Sensor fusion QA:** text + time‑series (audio/IMU) with query‑aware budgets.

Each benchmark will report **accuracy vs. token‑budget curves** and **hallucination/faithfulness** under compression.

---

## Roadmap

* [ ] MVP: registry + two encoders (text/layout) + simple top‑k bottleneck
* [ ] IB/RD proxies + target‑budget API
* [ ] STM v0 (append‑only, local FS) + retrieval hooks
* [ ] Orchestrator v0 (budget sweeps, retention probes)
* [ ] Example notebooks + tiny benchmarks
* [ ] Paper alignment: figures + ablations

---

## Research artifact

Place the draft here:

```
paper/Toward-N-Dimensional-LLMs-with-Information-Bottlenecks.pdf
```

(Optionally mirror to `docs/` and link from GitHub Pages.)

---

## Citing

If you use or build on this work, please cite the accompanying paper.

```bibtex
@misc{farrow2025ndllm,
  title        = {Toward N-Dimensional LLMs with Information Bottlenecks},
  author       = {Farrow, Joshua},
  year         = {2025},
  url          = {https://github.com/<you>/n-dimensional-llm},
  note         = {Preprint}
}
```

---

## Contributing

PRs and issues welcome. Please:

1. Open an issue describing the problem or proposal.
2. Include a minimal repro (dataset slice, config, expected behavior).
3. Target clean, typed code with tests where possible.

Code of conduct and contribution guide will live in `CONTRIBUTING.md`.

---

## License

TBD (recommendation: **Apache‑2.0** for permissive use with patent grant).

---

## Maintainer

**Joshua Farrow** — research & design.

For questions: open a GitHub issue or ping on X/Twitter `@jfarrow`.
