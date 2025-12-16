# mixture-llm

Combine multiple LLMs for better outputs.

## Install

```bash
pip install mixture-llm
```

## Usage

```python
from mixture_llm import Propose, Aggregate, run

pipeline = [
    Propose(["gpt-4", "claude-3", "llama-70b"]),
    Aggregate("gpt-4"),
]

result, history = await run(pipeline, "What is quantum computing?", my_client)
```

Your client function signature:
```python
async def my_client(model, messages, temp, max_tokens) -> tuple[str, int, int]:
    # Returns (response_text, input_tokens, output_tokens)
```

## Steps

**LLM steps:** `Propose`, `Synthesize`, `Aggregate`, `Refine`, `Rank`, `Vote`

**Transform steps:** `Shuffle`, `Dropout`, `Sample`, `Take`, `Filter`, `Map`

## Configuration

Override temp/max_tokens per-step:

```python
Propose(["gpt-4", "claude-3"], temp=0.9, max_tokens=4096)
```

Override prompts:

```python
Aggregate("gpt-4", prompt="Pick the best response and improve it.")
```

## Examples

```python
# Basic
[
    Propose(["gpt-4", "claude-3"]),
    Aggregate("gpt-4")
]

# With robustness
[
    Propose(["gpt-4", "claude-3", "llama-70b"]),
    Shuffle(),
    Dropout(0.2),
    Aggregate("gpt-4")
]

# Multi-layer
[
    Propose(["gpt-4", "claude-3"]),
    Synthesize(("gpt-4",)),
    Aggregate("gpt-4")
]
```

## License

MIT
