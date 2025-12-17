# mixture-llm

**Combine LLMs to beat the best single LLM.**

The Mixture-of-Agents architecture achieved **65.1% on AlpacaEval 2.0** using only open-source models, surpassing GPT-4o's 57.5%. This library gives you the building blocks to construct these pipelines.

## Why MoA?

LLMs exhibit a "collaborativeness" property: they generate better responses when given reference outputs from other models. MoA exploits this by passing outputs through multiple layers of LLM agents, with each layer refining the previous layer's work.

```python
from mixture_llm import Propose, Aggregate, run

pipeline = [
    Propose(["gpt-4o", "claude-sonnet-4-20250514", "llama-3.3-70b"]),
    Aggregate("gpt-4o"),
]

result, history = await run(pipeline, "What is quantum computing?", client)
```

## Key results from the research

| Configuration | AlpacaEval 2.0 |
|--------------|----------------|
| MoA (open-source only) | **65.1%** |
| MoA-Lite (2-layer) | 59.3% |
| GPT-4o standalone | 57.5% |

## Quick install

```bash
pip install mixture-llm
```

## Next steps

- [Getting Started](getting-started.md) — install and run your first pipeline
- [Examples](examples.md) — tested, runnable scripts for different providers
- [Pipelines](pipelines.md) — paper-accurate configurations
- [Steps Reference](steps.md) — all available pipeline steps
- [Client Examples](clients.md) — OpenAI, Anthropic, OpenRouter, Groq
