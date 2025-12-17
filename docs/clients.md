# Client Examples

Your client is an async function with this signature:

```python
async def client(model, messages, temp, max_tokens) -> tuple[str, int, int]:
    """
    Args:
        model: Model identifier string
        messages: List of {"role": str, "content": str} dicts
        temp: Sampling temperature
        max_tokens: Maximum response tokens

    Returns:
        Tuple of (response_text, input_tokens, output_tokens)
    """
```

## OpenAI

```python
from openai import AsyncOpenAI

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
```

## OpenAI + Anthropic (Multi-Provider)

Use two `AsyncOpenAI` instances with different base URLs:

```python
import os
from openai import AsyncOpenAI

openai_client = AsyncOpenAI()
anthropic_client = AsyncOpenAI(
    base_url="https://api.anthropic.com/v1/",
    api_key=os.environ["ANTHROPIC_API_KEY"],
)

async def multi_provider_client(model, messages, temp, max_tokens):
    client = anthropic_client if model.startswith("claude") else openai_client
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
```

**Example pipeline mixing providers:**

```python
from mixture_llm import Propose, Aggregate

pipeline = [
    Propose(["gpt-5.2", "claude-sonnet-4-5-20250514", "gpt-5.2-mini"]),
    Aggregate("claude-sonnet-4-5-20250514"),
]
```

## OpenRouter (All Models via One API)

[OpenRouter](https://openrouter.ai) provides access to hundreds of models through a single API:

```python
import os
from openai import AsyncOpenAI

client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)

async def openrouter_client(model, messages, temp, max_tokens):
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
```

**Together MoA models via OpenRouter:**

```python
PROPOSERS = [
    "qwen/qwen-2.5-72b-instruct",
    "meta-llama/llama-3.3-70b-instruct",
    "mistralai/mixtral-8x22b-instruct",
    "databricks/dbrx-instruct",
]

together_moa_openrouter = [
    Propose(PROPOSERS, temp=0.7, max_tokens=512),
    Synthesize(PROPOSERS, temp=0.7, max_tokens=512),
    Aggregate("qwen/qwen-2.5-72b-instruct"),
]
```

## Groq via LiteLLM (Free Tier)

[Groq](https://groq.com) offers free access to several models with blazing fast inference:

```python
from litellm import acompletion

async def groq_client(model, messages, temp, max_tokens):
    resp = await acompletion(
        model=f"groq/{model}",
        messages=messages,
        temperature=temp,
        max_tokens=max_tokens,
    )
    return (
        resp.choices[0].message.content,
        resp.usage.prompt_tokens,
        resp.usage.completion_tokens,
    )
```

**Free Groq models** (check [console.groq.com/docs/rate-limits](https://console.groq.com/docs/rate-limits) for current list):

```python
GROQ_FREE = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "qwen/qwen3-32b",
    "meta-llama/llama-4-scout-17b-16e-instruct",
]

free_moa = [
    Propose(GROQ_FREE, temp=0.7, max_tokens=512),
    Aggregate("llama-3.3-70b-versatile"),
]
```

**Self-MoA with Groq:**

```python
free_self_moa = [
    Propose(["llama-3.3-70b-versatile"] * 4, temp=0.7),
    Aggregate("llama-3.3-70b-versatile"),
]
```

!!! warning "Rate limits"

    Groq free tier has rate limits (30 RPM for most models). For production workloads, consider paid tiers or other providers.

## Together AI

```python
from openai import AsyncOpenAI

client = AsyncOpenAI(
    base_url="https://api.together.xyz/v1",
    api_key=os.environ["TOGETHER_API_KEY"],
)

async def together_client(model, messages, temp, max_tokens):
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
```

## Error Handling

The library handles errors gracefully—failed LLM calls are excluded from results:

```python
async def robust_client(model, messages, temp, max_tokens):
    try:
        # Your API call
        ...
    except Exception as e:
        raise  # Library catches this and records the error

# Errors appear in history
result, history = await run(pipeline, query, robust_client)
for step in history:
    for call in step["llm_calls"]:
        if "error" in call:
            print(f"{call['model']} failed: {call['error']}")
```

## Custom Retry Logic

Add retries for rate limits:

```python
import asyncio
from openai import AsyncOpenAI, RateLimitError

client = AsyncOpenAI()

async def retry_client(model, messages, temp, max_tokens):
    for delay in [1, 2, 4, 8]:
        try:
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
        except RateLimitError:
            await asyncio.sleep(delay)
    raise Exception(f"Rate limited after retries: {model}")
```
