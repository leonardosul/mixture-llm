# Examples

Runnable examples for different LLM providers.

## Setup

```bash
# Clone the repo
git clone https://github.com/yourusername/mixture-llm
cd mixture-llm

# Install with all provider dependencies
pip install -e ".[all]"

# Or install specific providers
pip install -e ".[openai]"
pip install -e ".[litellm]"
```

## Running examples

Each example requires the appropriate API key as an environment variable:

```bash
# OpenAI
export OPENAI_API_KEY=sk-...
python examples/openai_basic.py

# OpenAI + Anthropic
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
python examples/multi_provider.py

# OpenRouter
export OPENROUTER_API_KEY=sk-or-...
python examples/openrouter_moa.py

# Groq (free tier)
export GROQ_API_KEY=gsk_...
python examples/groq_free.py
python examples/with_history.py
```

## Examples

| Example | Provider | Description |
|---------|----------|-------------|
| [`openai_basic.py`](examples/openai_basic.py) | OpenAI | Simplest MoA with GPT-4o-mini |
| [`openai_self_moa.py`](examples/openai_self_moa.py) | OpenAI | Self-MoA (6 samples, one model) |
| [`multi_provider.py`](examples/multi_provider.py) | OpenAI + Anthropic | Mix GPT-4o and Claude |
| [`openrouter_moa.py`](examples/openrouter_moa.py) | OpenRouter | 3-layer Together MoA config |
| [`groq_free.py`](examples/groq_free.py) | Groq | Free tier, zero cost |
| [`with_history.py`](examples/with_history.py) | Groq | Inspect execution & costs |
