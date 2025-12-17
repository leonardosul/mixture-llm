# API Reference

## Core Function

### `run`

```python
async def run(
    pipeline: list[Any],
    query: str,
    client: Client
) -> tuple[str, list[dict[str, Any]]]
```

Execute a pipeline against a query.

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `pipeline` | `list` | List of pipeline steps |
| `query` | `str` | User query to process |
| `client` | `Client` | Async function to call LLMs |

**Returns:**

A tuple of `(result, history)`:

- `result`: Final response string (empty string if pipeline produces no output)
- `history`: List of step execution records

**History record structure:**

```python
{
    "step": str,              # Step class name ("Propose", "Aggregate", etc.)
    "outputs": list[str],     # Responses after this step
    "llm_calls": list[dict],  # Details of each LLM call
    "step_time": float,       # Seconds elapsed
}
```

**LLM call structure:**

```python
{
    "model": str,       # Model identifier
    "time": float,      # Call duration in seconds
    "in_tokens": int,   # Input tokens
    "out_tokens": int,  # Output tokens
    "error": str,       # Only present if call failed
}
```

---

## Client Protocol

```python
class Client(Protocol):
    def __call__(
        self,
        *,
        model: str,
        messages: list[Message],
        temp: float,
        max_tokens: int,
    ) -> Awaitable[tuple[str, int, int]]: ...
```

Your client must be an async callable that returns `(response_text, input_tokens, output_tokens)`.

**Message type:**

```python
class Message(TypedDict):
    role: str      # "system", "user", or "assistant"
    content: str   # Message content
```

---

## LLM Steps

### `Propose`

```python
class Propose(NamedTuple):
    agents: list[str]
    temp: float = 0.7
    max_tokens: int = 2048
```

Generate initial responses from multiple models in parallel.

### `Synthesize`

```python
class Synthesize(NamedTuple):
    agents: list[str]
    prompt: str = P_SYNTH
    temp: float = 0.7
    max_tokens: int = 2048
```

Each agent synthesizes all previous responses.

### `Aggregate`

```python
class Aggregate(NamedTuple):
    agent: str
    prompt: str = P_SYNTH
    temp: float = 0.7
    max_tokens: int = 2048
```

Single agent combines all responses into one.

### `Refine`

```python
class Refine(NamedTuple):
    agents: list[str]
    prompt: str = P_REFINE
    temp: float = 0.7
    max_tokens: int = 2048
```

Improve each response individually. Agents are cycled if fewer than responses.

### `Rank`

```python
class Rank(NamedTuple):
    agent: str
    n: int = 3
    prompt: str = P_RANK
    temp: float = 0.7
    max_tokens: int = 2048
```

Select top N responses by quality.

### `Vote`

```python
class Vote(NamedTuple):
    agent: str
    prompt: str = P_VOTE
    temp: float = 0.7
    max_tokens: int = 2048
```

Find consensus or select best answer.

---

## Transform Steps

### `Shuffle`

```python
class Shuffle(NamedTuple): ...
```

Randomize response order.

### `Dropout`

```python
class Dropout(NamedTuple):
    rate: float  # 0.0 to 1.0
```

Randomly drop responses with given probability.

### `Sample`

```python
class Sample(NamedTuple):
    n: int
```

Take random subset of N responses.

### `Take`

```python
class Take(NamedTuple):
    n: int
```

Keep first N responses.

### `Filter`

```python
class Filter(NamedTuple):
    fn: Callable[[str], bool]
```

Keep responses where `fn(response)` returns True.

### `Map`

```python
class Map(NamedTuple):
    fn: Callable[[str], str]
```

Transform each response with `fn(response)`.

---

## Constants

### Default Temperature

```python
DEFAULT_TEMP = 0.7
```

### Default Max Tokens

```python
DEFAULT_MAX_TOKENS = 2048
```

### Prompts

```python
P_SYNTH = """You have been provided with responses from various models to a query. \
Synthesize into a single, high-quality response. \
Critically evaluate—some may be biased or incorrect. \
Do not simply replicate; offer a refined, accurate reply."""

P_REFINE = "Improve this response:\n\n{text}\n\nOriginal query: {query}"

P_VOTE = """These responses answer the same question. \
Identify the consensus view shared by the majority. \
If no clear consensus, select the most accurate answer. \
Return only that answer, restated clearly."""

P_RANK = """Rank these responses by quality for the query: '{query}'

{responses}

Return the top {n} as comma-separated numbers (e.g., '3, 1, 5')."""
```

---

## Type Exports

```python
from mixture_llm import (
    # Core
    run,
    Message,
    Client,

    # LLM steps
    Propose,
    Synthesize,
    Aggregate,
    Refine,
    Rank,
    Vote,

    # Transform steps
    Shuffle,
    Dropout,
    Sample,
    Take,
    Filter,
    Map,

    # Constants
    DEFAULT_TEMP,
    DEFAULT_MAX_TOKENS,
    P_SYNTH,
    P_REFINE,
    P_VOTE,
    P_RANK,
)
```
