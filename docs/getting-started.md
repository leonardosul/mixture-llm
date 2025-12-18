# Getting Started

## Installation

```bash
pip install mixture-llm
```

## Basic usage

Every MoA pipeline needs three things:

1. **A pipeline** — a list of steps that process responses
2. **A query** — the user's question
3. **A client** — an async function that calls your LLM provider

```python
import asyncio
from mixture_llm import Propose, Aggregate, run

# Your client wraps your LLM provider
async def client(model, messages, temp, max_tokens):
    # Call your LLM here
    # Return (response_text, input_tokens, output_tokens)
    ...

# Define your pipeline
pipeline = [
    Propose(["gpt-5-nano-2025-08-07", "claude-sonnet-4-5"]),
    Aggregate("gpt-5-nano-2025-08-07"),
]

# Run it
async def main():
    result, history = await run(pipeline, "Explain quantum entanglement", client)
    print(result)

asyncio.run(main())
```

## Understanding the output

`run()` returns a tuple of `(result, history)`:

- **result** — the final synthesized response string
- **history** — a list of step records for debugging/analysis

Each history record contains:

```python
{
    "step": "Propose",           # Step type
    "outputs": ["...", "..."],   # Responses after this step
    "llm_calls": [...],          # Details of each LLM call
    "step_time": 2.34            # Seconds elapsed
}
```

## Minimal working example

Here's a complete example using OpenAI:

```python
import asyncio
import os
from openai import AsyncOpenAI
from mixture_llm import Propose, Aggregate, run

client = AsyncOpenAI()

async def openai_client(model, messages, temp, max_tokens):
    resp = await client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temp,
        max_tokens=max_tokens,
    )
    return (
        resp.choices[0].message.content,
        resp.usage.prompt_tokens,
        resp.usage.completion_tokens,
    )

pipeline = [
    Propose(["gpt-5-nano-2025-08-07", "gpt-5-nano-2025-08-07", "gpt-5-nano-2025-08-07"], temp=0.7),
    Aggregate("gpt-5-nano-2025-08-07"),
]

async def main():
    result, history = await run(
        pipeline,
        "What are the pros and cons of microservices?",
        openai_client,
    )
    print(result)

    # Check token usage
    total_in = sum(c["in_tokens"] for h in history for c in h["llm_calls"])
    total_out = sum(c["out_tokens"] for h in history for c in h["llm_calls"])
    print(f"\nTokens: {total_in} in, {total_out} out")

asyncio.run(main())
```

## Next steps

- See [Pipelines](pipelines.md) for paper-accurate configurations
- See [Client Examples](clients.md) for more provider setups
