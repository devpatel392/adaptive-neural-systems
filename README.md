# adaptive-neural-systems

Independent research into neural networks, deep learning, adaptive retrieval, graph-based reasoning, fine-tuning and continual learning.

## Research vision

Most "AI systems" today are assembled from pretrained models and prompt logic. This project asks a narrower and more testable question: **which parts of an intelligent retrieval-and-reasoning system benefit from being genuinely trained, and by how much?**

We build each component from first principles, train it on labels derived from real measurements, and compare it against simple baselines on held-out data. A component earns its place in the architecture only if it beats the baseline by a margin larger than seed-to-seed noise.

## Core objectives

1. **Understand neural networks from the ground up.** Implement and train networks from scratch (NumPy first, then PyTorch) before using higher-level libraries.
2. **Build a measurable retrieval stack.** Compare lexical, dense, hybrid and graph-based retrieval on the same question sets with the same metrics.
3. **Train an adaptive retrieval router.** Learn which retrieval strategy to use per query, supervised by labels that come from *actually running* each strategy and scoring its results. Labels are never hand-assigned question categories.
4. **Study learning over time.** Separate two different things that are often both called "learning":
   - *Weight updates*: gradient-based changes to model parameters (fine-tuning, LoRA, continual learning).
   - *Knowledge updates*: adding facts to a document store or knowledge graph. No parameters change.
5. **Apply the stack to real domains.** Scientific literature, finance and software engineering, evaluated with domain-appropriate metrics.

## Technical direction

| Area | Direction |
| --- | --- |
| Foundations | MLPs, backpropagation, optimisers and regularisation, written from scratch and then in PyTorch |
| Sequence models | A small Transformer trained from scratch on CPU-sized data |
| Retrieval | BM25, dense embeddings, hybrid fusion, GraphRAG |
| Graphs | Knowledge-graph construction, graph neural networks |
| Adaptation | LoRA/QLoRA on small open models, continual learning from verified feedback |
| Multimodal | Later phase, scoped once the text stack is evaluated |

See [docs/architecture.md](docs/architecture.md) for the system design and [docs/research-plan.md](docs/research-plan.md) for hypotheses, baselines and metrics.

## Development phases

| Phase | Focus | Status |
| --- | --- | --- |
| 0 | Repository foundation, tooling, research documentation | **Current** |
| 1 | Neural networks from scratch (NumPy MLP, then PyTorch) | Planned |
| 2 | Small Transformer trained from scratch | Planned |
| 3 | Retrieval baselines: BM25, dense, hybrid; evaluation harness | Planned |
| 4 | Knowledge graphs, GraphRAG and GNNs | Planned |
| 5 | Neural retrieval router trained on measured retrieval outcomes | Planned |
| 6 | Parameter-efficient fine-tuning (LoRA/QLoRA) | Planned |
| 7 | Continual learning from verified feedback | Planned |
| 8 | Multimodal extensions and domain applications | Planned |

A phase is complete only when its criteria in [docs/research-plan.md](docs/research-plan.md) are met.

## Hardware

Development targets a laptop with an Intel Core i5, 16 GB RAM and integrated graphics. All training is CPU-only, so models and datasets are sized to train in minutes, not days. Larger runs, if ever needed, will be planned explicitly.

## Local setup

Requires Python 3.11 or newer. Run the following from the repository root in PowerShell:

```powershell
# 1. Create and activate a virtual environment (isolates this project's packages)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Install the package in editable mode with development tools (pytest, ruff)
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"

# 3. Run checks
python -m pytest
python -m ruff check .
python -m ruff format --check .
```

If PowerShell blocks the activation script, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once.

### Later phases: CPU-only PyTorch

Not needed for Phase 0. When Phase 1 moves to PyTorch, install the CPU build explicitly so pip does not pull the much larger CUDA build:

```powershell
python -m pip install -e ".[dev,ml]"
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## Repository layout

```
src/adaptive_neural_systems/   library code (importable package)
tests/                         pytest test suite
experiments/                   one folder per experiment: config, script, results summary
docs/                          architecture and research plan
CLAUDE.md                      conventions for AI-assisted development
```
