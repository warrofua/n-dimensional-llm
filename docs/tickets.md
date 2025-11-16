# Ticket Backlog

Centralised backlog capturing the next implementation pushes required to reach the ND-LLM architecture described in the design docs.

## T1 – Field Registry & Canonical Synchronisation Layer

- **Goal:** Promote the ad-hoc field munging (e.g., `benchmarks/cord.py`, `nd_llm/model.py`) into a reusable registry that ingests arbitrary fields, aligns them into a canonical coordinate space (layout UV, timestamps, spans), and surfaces consistent tensors to the rest of the stack.
- **Scope:** Introduce a `FieldRegistry`/`FieldAdapter` module exposing encoder, alignment, and projection hooks; wire the canonical cell builder into `nd_llm.model.CanonicalCellAggregator`; add validation utilities under `nd_llm.registry`.
- **Acceptance:** Unit tests covering registration/alignment, docs describing how to register new fields, benchmarks updated to consume the registry.

## T2 – Mutual-Information-Aware Token Bottleneck

- **Goal:** Replace the FIFO/heuristic bottleneck with a learnable scorer that maximises an MI proxy per cell and supports variable-rate allocation, matching the “Mutual Information per Token” principle.
- **Scope:** Extend `nd_llm/bottleneck` with a scorer (InfoNCE/MINE surrogate), expose APIs for dynamic target budgets, integrate with the orchestrator’s budget allocator, add ablations showing R–D gains.
- **Acceptance:** Benchmark plots showing the MI-aware bottleneck dominating the current selector at equal token budgets; telemetry exposing per-token MI estimates.

## T3 – Semantic Tensor Memory & Constraint Integration

- **Goal:** Upgrade STM from an append-only log to a holographic tensor memory governed by IB policies, and integrate a neuro-symbolic constraint layer (LTN/TensorLog style) inside the reasoning loop.
- **Scope:** Enhance `nd_llm/stm` with write/read policies, add compression metrics, expose APIs for constraint modules; teach the orchestrator (`nd_llm/orchestration`) to route through STM and constraints.
- **Acceptance:** End-to-end example showing STM-assisted reasoning with constraints enabled, plus ablations (disable STM/constraints) demonstrating fidelity drops.

## T4 – Rate–Distortion & Fano Audit CLI

- **Goal:** Provide tooling to empirically validate the theory: sweep token budgets for text-only vs. N-D inputs, plot R–D curves, and compute Fano-consistent error bounds.
- **Scope:** New script/notebook under `scripts/` or `benchmarks/` to run both configurations, log metrics, and render plots; integrate with README/docs.
- **Acceptance:** Stored plot artifacts, README section linking to them, automated test (CI-safe subset) to ensure the CLI still runs.

## T5 – Documentation & Story Alignment *(done)*

- **Goal:** Keep the written narrative in lockstep with the implementation.
- **Scope:** Update `README.md` (bottleneck knobs, STM/superposition usage, rate–distortion CLI), `docs/the-tensor-is-the-message.md` (highlight empirical R–D tooling), and link the new registry/constraint features.
- **Acceptance:** Docs cite concrete modules and usage snippets; roadmap reflects the completed milestones.

## T6 – Dataset Expansion & Evaluation Coverage *(done)*

- **Goal:** Extend beyond the synthetic + CORD pairing once the registry exists, exercising field synchrony on at least one additional multi-field dataset (e.g., chart QA, timeline QA).
- **Scope:** Added the ChartQA harness (`benchmarks/chartqa.py`) with registry-aware field adapters, a bundled sample (`benchmarks/data/chartqa_sample.jsonl`), tests, and README instructions for plugging in the full dataset.
- **Acceptance:** New benchmark entry with tests, plus guidance in README/docs.

## T7 – Ollama LLM Harness for Local Testing *(done)*

- **Goal:** Exercise the encoder/orchestrator loop against a real LLM served locally via Ollama (e.g., `llama3.1:8b`) so developers can verify the end-to-end stack on Mac hardware.
- **Scope:** Added `scripts/ollama_harness.py` (with `--dry-run` mode), README instructions, and automated tests covering the prompt generation for both CORD and ChartQA samples.
- **Acceptance:** Documented instructions for using the harness, and a mock-tested path to keep CI green.
