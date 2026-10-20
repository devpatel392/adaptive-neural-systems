# Architecture

This document describes the system we intend to build and states clearly what exists today. Anything not listed under **Current implementation** is a plan, not a result.

## Current implementation (Phase 1, in progress)

| Component | Location | Purpose |
| --- | --- | --- |
| Package skeleton | `src/adaptive_neural_systems/` | Importable library using the `src` layout |
| Seed control | `src/adaptive_neural_systems/reproducibility.py` | `set_seed()` seeds Python, and NumPy/PyTorch when installed |
| Digits data splits | `src/adaptive_neural_systems/data.py` | Fixed, stratified 60/20/20 train/val/test split of scikit-learn's digits; pixels scaled to [0, 1] |
| Test suite | `tests/` | pytest, configured in `pyproject.toml` |
| Tooling | `pyproject.toml` | Packaging, pytest and ruff configuration |

No models, retrieval pipelines, graphs or training code exist yet. Experiment E1.1 is being built step by step.

### Why a `src` layout

With the package under `src/`, Python cannot import it by accident from the repository root. Tests therefore exercise the package as it is configured, which catches packaging mistakes early. `pytest` is configured with `pythonpath = ["src"]` so tests also run without an editable install.

### Dependency policy

Phase 0 has **zero runtime dependencies**. Every dependency is added in the phase that needs it, with its reason recorded here.

| Group | Packages | Reason | Phase |
| --- | --- | --- | --- |
| `dev` | pytest, ruff | Testing; linting and formatting in one fast tool | 0 |
| `ml` | numpy, scikit-learn (pulls in scipy, joblib) | Array maths; digits dataset, stratified splits, logistic-regression baseline | 1 |
| `plot` | matplotlib | Learning curves (not yet installed) | 1 |
| (manual) | torch (CPU wheel) | Autograd and layers once from-scratch versions are understood | 1+ |

## Proposed system architecture

```
                 ┌──────────────────────────────┐
     query ───►  │  Query encoder (trained)     │
                 └──────────────┬───────────────┘
                                ▼
                 ┌──────────────────────────────┐
                 │  Adaptive retrieval router   │  learns: query -> best strategy
                 └───┬──────────┬──────────┬────┘
                     ▼          ▼          ▼
                 Lexical     Dense      Graph
                 (BM25)    (embeddings) (KG traversal / GNN)
                     └──────────┬──────────┘
                                ▼
                 ┌──────────────────────────────┐
                 │  Fusion + reranking          │
                 └──────────────┬───────────────┘
                                ▼
                 ┌──────────────────────────────┐
                 │  Reader / generator          │  small LM, optionally LoRA-adapted
                 └──────────────┬───────────────┘
                                ▼
                 answer + cited evidence ──► verification ──► feedback store
                                                                   │
                         (verified feedback only) ◄────────────────┘
                         ├─► knowledge update: add to KG / index
                         └─► weight update: continual fine-tuning
```

### Neural network components

- **From-scratch MLP** (Phase 1). Forward pass, backpropagation and SGD/Adam implemented manually in NumPy and checked against numerical gradients. This is the foundation for understanding every later component.
- **Small Transformer** (Phase 2). Token embeddings, positional encoding, multi-head self-attention, layer norm and residual connections, trained from scratch on a small corpus on CPU.
- **Query encoder** (Phase 3+). Produces dense vectors for retrieval. The baseline is a small pretrained sentence encoder; a from-scratch encoder is studied only if it can be evaluated fairly.

### Retrieval and graph components

- **Lexical retrieval**: BM25. A strong, cheap baseline that every neural method must beat.
- **Dense retrieval**: nearest neighbour search over embeddings.
- **Hybrid retrieval**: score or rank fusion (e.g. reciprocal rank fusion) of lexical and dense results.
- **Knowledge graph**: entities and relations extracted from the corpus, with provenance for every edge.
- **GraphRAG**: retrieval by graph traversal and community summaries.
- **Graph neural networks**: message passing over the knowledge graph for link prediction and node relevance.

### Adaptive retrieval router

The router is a small classifier or regressor that predicts, for each query, which retrieval strategy will perform best.

Its **labels must come from measurements**. For every training query we run each strategy, score the retrieved results against ground-truth relevance (e.g. recall@k), and use that outcome as the target. We never hand-assign labels such as "this is a multi-hop question, so use the graph". Those labels would encode our assumptions, and the router would only learn to reproduce them.

Baselines the router must beat:

1. Always use the single best strategy.
2. Always use hybrid fusion.
3. A simple heuristic, such as query length or entity count.
4. Oracle upper bound: the best strategy chosen per query with hindsight. This is not a baseline to beat; it shows how much routing could ever gain.

### Fine-tuning and learning components

- **LoRA / QLoRA**. Low-rank adapters let us fine-tune a small open model by training a few million parameters instead of billions. Feasibility on 16 GB RAM, CPU-only, must be measured before committing to a model size.
- **Continual learning**. Updating weights over time from verified feedback while measuring forgetting on earlier tasks.

#### Knowledge updates vs. weight updates

| | Knowledge update | Weight update |
| --- | --- | --- |
| What changes | Index, document store or KG contents | Model parameters θ |
| Mechanism | Insert or delete records | Gradient descent: θ ← θ − η∇L(θ) |
| Reversible | Yes, by deleting the record | Not exactly; needs checkpoints |
| Risk | Stale or wrong facts | Catastrophic forgetting, overfitting |
| Evidence it worked | Fact is retrievable | Held-out metric improves, old tasks hold |

Reports from this project always state which of the two happened. Adding a fact to the graph is not "the model learned".

## Future plans (not implemented)

Everything under "Proposed system architecture" above. Multimodal inputs, domain applications, any user interface, databases and agent orchestration are explicitly out of scope until the text pipeline has been evaluated.
