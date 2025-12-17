# Examples

The [`examples/`](https://github.com/leonardosul/mixture-llm/tree/main/examples) directory contains tested, runnable scripts demonstrating mixture-llm with different LLM providers and configurations.

## Quick Reference

| Example | Provider | Key Concepts | Best For |
|---------|----------|--------------|----------|
| [openai_basic.py](#openai-basic) | OpenAI | Propose, Aggregate | Getting started |
| [openai_self_moa.py](#self-moa) | OpenAI | Self-MoA, single model sampling | Quality optimization |
| [multi_provider.py](#multi-provider) | OpenAI + Anthropic | Provider routing, Shuffle | Production setups |
| [openrouter_moa.py](#openrouter-moa) | OpenRouter | 3-layer MoA, Synthesize | Paper reproduction |
| [groq_free.py](#groq-free) | Groq | Free tier, Dropout, LiteLLM | Experimentation |
| [with_history.py](#history-inspection) | Groq | History inspection, Rank | Debugging & analysis |

## Setup

### Install Dependencies

```bash
pip install mixture-llm[examples]
```

Or install specific providers:

```bash
pip install mixture-llm[openai]    # OpenAI + Anthropic
pip install mixture-llm[litellm]   # Groq via LiteLLM
```

### Set API Keys

```bash
export OPENAI_API_KEY=sk-...           # OpenAI examples
export ANTHROPIC_API_KEY=sk-ant-...    # Multi-provider example
export OPENROUTER_API_KEY=sk-or-...    # OpenRouter example
export GROQ_API_KEY=gsk_...            # Groq examples (free)
```

---

## OpenAI Basic

**File:** [`openai_basic.py`](https://github.com/leonardosul/mixture-llm/blob/main/examples/openai_basic.py)

The simplest MoA implementation. Start here to understand the core pattern.

```python
from mixture_llm import Propose, Aggregate, run

pipeline = [
    Propose(["gpt-4o-mini"] * 3, temp=0.7),  # 3 proposals via temperature
    Aggregate("gpt-4o-mini"),                 # Combine into final answer
]

result, history = await run(pipeline, query, openai_client)
```

**What it demonstrates:**

- Basic 2-layer MoA (Propose → Aggregate)
- Using temperature (0.7) to create diverse responses from the same model
- Client function that returns `(text, input_tokens, output_tokens)`
- Token usage and timing analysis

**Run:**

```bash
export OPENAI_API_KEY=sk-...
python examples/openai_basic.py
```

---

## Self-MoA

**File:** [`openai_self_moa.py`](https://github.com/leonardosul/mixture-llm/blob/main/examples/openai_self_moa.py)

Based on [Li et al. (2025)](https://arxiv.org/abs/2502.00674): sampling one great model multiple times outperforms diverse mediocre models by **+6.6%** on AlpacaEval 2.0.

```python
pipeline = [
    Propose(["gpt-4o"] * 6, temp=0.7, max_tokens=512),  # 6 samples, one model
    Aggregate("gpt-4o", max_tokens=1024),
]
```

**What it demonstrates:**

- Self-MoA configuration from the research paper
- Temperature creates diversity even with identical models
- Inspecting individual proposals in the history
- Quality over quantity approach

**Run:**

```bash
export OPENAI_API_KEY=sk-...
python examples/openai_self_moa.py
```

---

## Multi-Provider

**File:** [`multi_provider.py`](https://github.com/leonardosul/mixture-llm/blob/main/examples/multi_provider.py)

Mix models from different providers in a single pipeline, combining the strengths of GPT-4o and Claude.

```python
# Route to appropriate provider based on model name
openai_client = AsyncOpenAI()
anthropic_client = AsyncOpenAI(
    base_url="https://api.anthropic.com/v1/",
    api_key=os.environ["ANTHROPIC_API_KEY"],
)

async def multi_provider_client(model, messages, temp, max_tokens):
    client = anthropic_client if model.startswith("claude") else openai_client
    # ... make API call

pipeline = [
    Propose(["gpt-4o", "claude-sonnet-4-20250514", "gpt-4o-mini"], temp=0.7, max_tokens=512),
    Shuffle(),  # Prevent position bias
    Aggregate("claude-sonnet-4-20250514", max_tokens=1024),
]
```

**What it demonstrates:**

- Provider routing based on model name prefix
- Using Anthropic via OpenAI-compatible API
- `Shuffle()` step to randomize response order (prevents position bias)
- Cross-provider aggregation

**Run:**

```bash
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
python examples/multi_provider.py
```

---

## OpenRouter MoA

**File:** [`openrouter_moa.py`](https://github.com/leonardosul/mixture-llm/blob/main/examples/openrouter_moa.py)

3-layer MoA approximating the benchmark-winning configuration from [Wang et al. (2024)](https://arxiv.org/abs/2406.04692) that achieved **65.1% on AlpacaEval 2.0**.

```python
PROPOSERS = [
    "qwen/qwen-2.5-72b-instruct",
    "meta-llama/llama-3.3-70b-instruct",
    "mistralai/mixtral-8x22b-instruct",
    "databricks/dbrx-instruct",
]

pipeline = [
    Propose(PROPOSERS, temp=0.7, max_tokens=512),      # Layer 1: Independent proposals
    Synthesize(PROPOSERS, temp=0.7, max_tokens=512),   # Layer 2: Each model sees all
    Aggregate("qwen/qwen-2.5-72b-instruct", max_tokens=1024),  # Layer 3: Final synthesis
]
```

**What it demonstrates:**

- Full 3-layer MoA architecture (Propose → Synthesize → Aggregate)
- `Synthesize` step where each model sees all previous responses
- OpenRouter for unified access to multiple providers
- Model namespacing (`provider/model-name` format)
- Timing breakdown by layer

**Run:**

```bash
export OPENROUTER_API_KEY=sk-or-...
python examples/openrouter_moa.py
```

---

## Groq Free

**File:** [`groq_free.py`](https://github.com/leonardosul/mixture-llm/blob/main/examples/groq_free.py)

Zero-cost experimentation using Groq's free tier with fast inference. Demonstrates both standard MoA and Self-MoA.

```python
from litellm import acompletion

async def groq_client(model, messages, temp, max_tokens):
    resp = await acompletion(
        model=f"groq/{model}",  # LiteLLM prefix
        messages=messages,
        temperature=temp,
        max_tokens=max_tokens,
    )
    return (...)

GROQ_FREE = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "qwen/qwen3-32b",
    "meta-llama/llama-4-scout-17b-16e-instruct",
]

# Standard MoA with robustness
pipeline = [
    Propose(GROQ_FREE, temp=0.7, max_tokens=512),
    Shuffle(),
    Dropout(0.2),  # 20% random dropout
    Aggregate("llama-3.3-70b-versatile", max_tokens=1024),
]

# Self-MoA variant (also included)
self_moa_pipeline = [
    Propose(["llama-3.3-70b-versatile"] * 4, temp=0.7, max_tokens=512),
    Aggregate("llama-3.3-70b-versatile", max_tokens=1024),
]
```

**What it demonstrates:**

- LiteLLM integration with `groq/` prefix
- `Dropout(0.2)` step for robustness (randomly drops 20% of responses)
- Free tier models from Groq
- Running both standard and Self-MoA in one script
- Execution details per step

**Run:**

```bash
export GROQ_API_KEY=gsk_...
python examples/groq_free.py
```

!!! note "Rate Limits"
    Groq free tier has rate limits (typically 30 RPM). Check [console.groq.com/docs/rate-limits](https://console.groq.com/docs/rate-limits) for current limits.

---

## History Inspection

**File:** [`with_history.py`](https://github.com/leonardosul/mixture-llm/blob/main/examples/with_history.py)

Deep inspection of pipeline execution for debugging, analysis, and cost tracking.

```python
pipeline = [
    Propose(["llama-3.3-70b-versatile"] * 4, temp=0.7, max_tokens=256),
    Shuffle(),
    Rank("llama-3.3-70b-versatile", n=2),  # Keep top 2 by quality
    Aggregate("llama-3.3-70b-versatile", max_tokens=512),
]
```

**What it demonstrates:**

- `Rank` step that uses an LLM to select the top N responses by quality
- Detailed history inspection with `print_history()` helper
- Per-step timing and token counts
- Output previews at each stage
- JSON export for external analysis (`history.json`)
- Error tracking (✓ for success, ✗ for failures)

**History structure:**

```python
{
    "step": "Propose",           # Step type name
    "outputs": ["...", "..."],   # Responses after this step
    "llm_calls": [
        {
            "model": "llama-3.3-70b-versatile",
            "time": 2.34,
            "in_tokens": 150,
            "out_tokens": 280,
            "error": None,       # Present only if error occurred
        }
    ],
    "step_time": 2.34            # Total step execution time
}
```

**Run:**

```bash
export GROQ_API_KEY=gsk_...
python examples/with_history.py
```

---

## Common Patterns

### Client Function Template

All examples follow this signature:

```python
async def client(model, messages, temp, max_tokens) -> tuple[str, int, int]:
    """
    Call your LLM provider.

    Args:
        model: Model identifier string
        messages: List of message dicts with 'role' and 'content'
        temp: Temperature for sampling (0.0-1.0)
        max_tokens: Maximum tokens in response

    Returns:
        tuple of (response_text, input_tokens, output_tokens)
    """
    ...
```

### Token and Time Tracking

```python
result, history = await run(pipeline, query, client)

total_in = sum(c["in_tokens"] for h in history for c in h["llm_calls"])
total_out = sum(c["out_tokens"] for h in history for c in h["llm_calls"])
total_time = sum(h["step_time"] for h in history)

print(f"Tokens: {total_in:,} in, {total_out:,} out")
print(f"Time: {total_time:.2f}s")
```

### Error Handling

```python
for step in history:
    for call in step["llm_calls"]:
        if "error" in call:
            print(f"Error in {call['model']}: {call['error']}")
```

---

## Next Steps

- [Getting Started](getting-started.md) — Installation and basic concepts
- [Pipelines](pipelines.md) — Paper-accurate configurations
- [Steps Reference](steps.md) — All available pipeline steps
- [Client Examples](clients.md) — More provider integrations
