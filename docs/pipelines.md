# Pipelines

This page documents paper-accurate pipeline configurations that reproduce benchmark results.

## Together MoA (65.1% AlpacaEval)

The benchmark-winning configuration from [Wang et al. (2024)](https://arxiv.org/abs/2406.04692): 3 layers, 6 diverse proposers, Qwen aggregator.

```python
from mixture_llm import Propose, Synthesize, Aggregate

PROPOSERS = [
    "wizardlm-2-8x22b",
    "qwen1.5-110b-chat",
    "qwen1.5-72b-chat",
    "llama-3-70b-instruct",
    "mixtral-8x22b-instruct",
    "dbrx-instruct",
]

together_moa = [
    Propose(PROPOSERS, temp=0.7, max_tokens=512),
    Synthesize(PROPOSERS, temp=0.7, max_tokens=512),
    Synthesize(PROPOSERS, temp=0.7, max_tokens=512),
    Aggregate("qwen1.5-110b-chat"),
]
```

!!! info "How it works"

    1. **Layer 1 (Propose)**: All 6 models independently answer the query
    2. **Layer 2 (Synthesize)**: Each model sees all 6 responses and synthesizes
    3. **Layer 3 (Synthesize)**: Another round of synthesis
    4. **Final (Aggregate)**: Qwen combines everything into the final answer

## MoA-Lite (59.3% AlpacaEval)

Cost-optimized 2-layer variant—still beats GPT-4o while using ~3.5x fewer tokens.

```python
moa_lite = [
    Propose(PROPOSERS, temp=0.7, max_tokens=512),
    Synthesize(PROPOSERS, temp=0.7, max_tokens=512),
    Aggregate("qwen1.5-72b-chat"),
]
```

## MoA with GPT-5 Aggregator (65.7% AlpacaEval)

The highest-scoring configuration uses GPT-5.2 as the final aggregator:

```python
moa_gpt5 = [
    Propose(PROPOSERS, temp=0.7, max_tokens=512),
    Synthesize(PROPOSERS, temp=0.7, max_tokens=512),
    Synthesize(PROPOSERS, temp=0.7, max_tokens=512),
    Aggregate("gpt-5.2-chat-latest"),
]
```

!!! tip "Aggregator selection"

    Research shows aggregator quality has **2x more impact** on final performance than proposer quality. Invest in your aggregator model.

## Self-MoA (+6.6% over standard MoA)

[Li et al. (2025)](https://arxiv.org/abs/2502.00674) showed that sampling one top model multiple times can outperform diverse model mixtures by 6.6% on AlpacaEval.

```python
# Same model, multiple samples via temperature
# Note: Use models that support temperature (not GPT-5.2)
self_moa = [
    Propose(["gpt-4.1"] * 6, temp=0.7),
    Aggregate("gpt-4.1"),
]
```

!!! note "When to use Self-MoA"

    Use Self-MoA when you have access to one very strong model. The key insight: mixing different LLMs often lowers average quality. A single excellent model sampled multiple times can beat a diverse group of weaker models.

## Self-MoA Sequential

For context-limited scenarios, Self-MoA-Seq uses a sliding window approach:

```python
# Iterative refinement for long contexts
self_moa_seq = [
    Propose(["gpt-4.1"] * 3, temp=0.7),
    Aggregate("gpt-4.1"),
    Propose(["gpt-4.1"] * 3, temp=0.7),
    Aggregate("gpt-4.1"),
]
```

## Robust MoA

Add shuffle and dropout to prevent positional bias and improve diversity:

```python
from mixture_llm import Shuffle, Dropout

robust_moa = [
    Propose(["gpt-5.2-chat-latest", "claude-sonnet-4-5", "llama-3.3-70b", "gemini-2.5-flash"]),
    Shuffle(),
    Dropout(0.2),
    Aggregate("gpt-5.2-chat-latest"),
]
```

## Rank-then-Aggregate

Use an LLM to select the best responses before aggregating:

```python
from mixture_llm import Rank

rank_aggregate = [
    Propose(["gpt-5.2-chat-latest", "claude-sonnet-4-5", "llama-3.3-70b", "gemini-2.5-flash", "gpt-4.1-mini"]),
    Rank("gpt-5.2-chat-latest", n=3),  # Keep top 3
    Aggregate("gpt-5.2-chat-latest"),
]
```

## Vote (Consensus Selection)

For tasks with clear correct answers, use voting to find consensus:

```python
from mixture_llm import Vote

vote_pipeline = [
    Propose(["gpt-5.2-chat-latest", "claude-sonnet-4-5", "llama-3.3-70b"], temp=0.3),
    Vote("gpt-5.2-chat-latest"),
]
```

## Refine Pipeline

Improve each response individually before aggregating:

```python
from mixture_llm import Refine

refine_pipeline = [
    Propose(["gpt-4.1-mini", "claude-haiku-4", "llama-3.1-8b"]),
    Refine(["gpt-5.2-chat-latest"]),  # GPT-5.2 improves each response
    Aggregate("gpt-5.2-chat-latest"),
]
```

## Configuration Guidelines

| Use case | Recommended pipeline |
|----------|---------------------|
| Maximum quality | Together MoA (3 layers, 6 proposers) |
| Cost-effective | MoA-Lite (2 layers) |
| Single top model available | Self-MoA |
| Factual/objective tasks | Vote |
| Need diversity | Robust MoA with Shuffle + Dropout |
| Context-limited | Self-MoA Sequential |

## Layer Depth Performance

Research shows diminishing returns beyond 3 layers:

| Layers | AlpacaEval 2.0 |
|--------|----------------|
| 1 | ~44% |
| 2 | ~61% |
| 3 | ~65% |
| 4 | ~66% |

**Recommendation**: Use 3 layers for quality-critical applications, 2 layers for cost-sensitive ones.
