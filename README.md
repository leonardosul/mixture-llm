# mixture-llm

Combine LLMs to beat the best single LLM.

The Mixture-of-Agents architecture achieved **65.1% on AlpacaEval 2.0** using only open-source models—surpassing GPT-4o's 57.5%. This library gives you the building blocks to construct these pipelines.

## Install

```bash
pip install mixture-llm
```

## Quick start

```python
from mixture_llm import Propose, Aggregate, run

pipeline = [
    Propose(["gpt-4o", "claude-sonnet-4-20250514", "llama-3.3-70b"]),
    Aggregate("gpt-4o"),
]

result, history = await run(pipeline, "What is quantum computing?", my_client)
```

## Paper-accurate pipelines

### Together MoA (65.1% AlpacaEval)

The benchmark-winning configuration from [Wang et al. (2024)](https://arxiv.org/abs/2406.04692): 3 layers, 6 diverse proposers, Qwen aggregator.

```python
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

### MoA-Lite (59.3% AlpacaEval)

Cost-optimized 2-layer variant—still beats GPT-4o.

```python
moa_lite = [
    Propose(PROPOSERS, temp=0.7, max_tokens=512),
    Synthesize(PROPOSERS, temp=0.7, max_tokens=512),
    Aggregate("qwen1.5-72b-chat"),
]
```

### Self-MoA (+6.6% over standard MoA)

[Li et al. (2025)](https://arxiv.org/abs/2502.00674) showed that sampling one top model multiple times can outperform diverse model mixtures.

```python
# Same model, multiple samples via temperature
self_moa = [
    Propose(["gpt-4o"] * 6, temp=0.7),
    Aggregate("gpt-4o"),
]
```

### With robustness (shuffle + dropout)

Prevents positional bias and improves diversity.

```python
robust_moa = [
    Propose(["gpt-4o", "claude-sonnet", "llama-70b", "gemini-pro"]),
    Shuffle(),
    Dropout(0.2),
    Aggregate("gpt-4o"),
]
```

## Steps

**LLM steps** — call models:
- `Propose(agents)` — generate initial responses in parallel
- `Synthesize(agents)` — each agent synthesizes all previous outputs
- `Aggregate(agent)` — single model combines everything into final output
- `Refine(agents)` — improve each response individually
- `Rank(agent, n)` — select top n responses by quality
- `Vote(agent)` — pick consensus answer

**Transform steps** — manipulate responses:
- `Shuffle()` — randomize order (prevents position bias)
- `Dropout(rate)` — randomly drop responses (improves robustness)
- `Sample(n)` — random subset
- `Take(n)` — first n responses
- `Filter(fn)` — keep responses matching predicate
- `Map(fn)` — transform each response

## Configuration

Every LLM step accepts `temp` and `max_tokens`:

```python
Propose(["gpt-4o", "claude-sonnet"], temp=0.9, max_tokens=4096)
```

Override the synthesis prompt:

```python
Aggregate("gpt-4o", prompt="Pick the single best response and return it verbatim.")
```

## Client examples

Your client is an async function with this signature:

```python
async def client(model, messages, temp, max_tokens) -> tuple[str, int, int]:
    # Returns (response_text, input_tokens, output_tokens)
```

### OpenAI SDK (OpenAI + Anthropic models)

```python
from openai import AsyncOpenAI

openai_client = AsyncOpenAI()
anthropic_client = AsyncOpenAI(
    base_url="https://api.anthropic.com/v1/",
    api_key=os.environ["ANTHROPIC_API_KEY"],
)

async def multi_provider_client(model, messages, temp, max_tokens):
    client = anthropic_client if model.startswith("claude") else openai_client
    resp = await client.chat.completions.create(
        model=model, messages=messages, temperature=temp, max_tokens=max_tokens
    )
    return resp.choices[0].message.content, resp.usage.prompt_tokens, resp.usage.completion_tokens

# Mix providers in one pipeline
pipeline = [
    Propose(["gpt-4o", "claude-sonnet-4-20250514", "gpt-4o-mini"]),
    Aggregate("claude-sonnet-4-20250514"),
]
```

### OpenRouter (access all models via one API)

```python
from openai import AsyncOpenAI

client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)

async def openrouter_client(model, messages, temp, max_tokens):
    resp = await client.chat.completions.create(
        model=model, messages=messages, temperature=temp, max_tokens=max_tokens
    )
    return resp.choices[0].message.content, resp.usage.prompt_tokens, resp.usage.completion_tokens

# Together MoA models via OpenRouter
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

### Groq via LiteLLM (free tier)

Groq offers free access to several models. Great for experimentation.

```python
from litellm import acompletion

async def groq_client(model, messages, temp, max_tokens):
    resp = await acompletion(
        model=f"groq/{model}", messages=messages, temperature=temp, max_tokens=max_tokens
    )
    return resp.choices[0].message.content, resp.usage.prompt_tokens, resp.usage.completion_tokens

# Free Groq models (check console.groq.com/docs/rate-limits for current list)
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

# Self-MoA with Groq (single model, multiple samples)
free_self_moa = [
    Propose(["llama-3.3-70b-versatile"] * 4, temp=0.7),
    Aggregate("llama-3.3-70b-versatile"),
]
```

## Examples

The [`examples/`](examples/) folder contains runnable scripts for different providers:

| Example | Provider | Description |
|---------|----------|-------------|
| [`openai_basic.py`](examples/openai_basic.py) | OpenAI | Simplest MoA with GPT-4o-mini |
| [`openai_self_moa.py`](examples/openai_self_moa.py) | OpenAI | Self-MoA (6 samples, one model) |
| [`multi_provider.py`](examples/multi_provider.py) | OpenAI + Anthropic | Mix GPT-4o and Claude |
| [`openrouter_moa.py`](examples/openrouter_moa.py) | OpenRouter | 3-layer Together MoA config |
| [`groq_free.py`](examples/groq_free.py) | Groq | Free tier, zero cost |
| [`with_history.py`](examples/with_history.py) | Groq | Inspect execution & costs |

```bash
# Install dependencies and run
pip install -e ".[examples]"
export OPENAI_API_KEY=sk-...
python examples/openai_basic.py
```

## Key findings from the research

- **Aggregator quality matters 2x more than proposer quality** — invest in your final model
- **3 layers is the sweet spot** — diminishing returns beyond this
- **Diversity vs quality tradeoff** — Self-MoA shows a single great model can beat diverse mediocre ones
- **6 proposers optimal** — gains diminish after this point

## References

- Wang et al. "Mixture-of-Agents Enhances Large Language Model Capabilities" (2024) — [arXiv:2406.04692](https://arxiv.org/abs/2406.04692)
- Li et al. "Rethinking Mixture-of-Agents: Is Mixing Different Large Language Models Beneficial?" (2025) — [arXiv:2502.00674](https://arxiv.org/abs/2502.00674)

## License

MIT
