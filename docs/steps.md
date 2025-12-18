# Steps Reference

Pipeline steps fall into two categories: **LLM steps** that call models, and **transform steps** that manipulate responses without LLM calls.

## LLM Steps

### Propose

Generate initial responses in parallel from multiple models.

```python
Propose(agents, temp=0.7, max_tokens=2048)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `agents` | `list[str]` | required | Model names to query |
| `temp` | `float` | 0.7 | Sampling temperature |
| `max_tokens` | `int` | 2048 | Maximum response length |

**Behavior**: Each agent receives the raw user query with no system prompt. All calls run in parallel.

```python
# Diverse models
Propose(["gpt-5.2-chat-latest", "claude-sonnet-4-5", "llama-3.3-70b"])

# Self-MoA: same model, multiple samples
Propose(["gpt-5.2-chat-latest"] * 6, temp=0.7)
```

---

### Synthesize

Each agent synthesizes all previous responses into a new response.

```python
Synthesize(agents, prompt=P_SYNTH, temp=0.7, max_tokens=2048)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `agents` | `list[str]` | required | Model names to query |
| `prompt` | `str` | synthesis prompt | System prompt for synthesis |
| `temp` | `float` | 0.7 | Sampling temperature |
| `max_tokens` | `int` | 2048 | Maximum response length |

**Behavior**: Each agent sees all previous responses and the original query. Produces N new responses (one per agent).

```python
# Standard MoA layer
Synthesize(["gpt-5.2-chat-latest", "claude-sonnet-4-5", "llama-3.3-70b"])
```

---

### Aggregate

Single model combines all responses into one final output.

```python
Aggregate(agent, prompt=P_SYNTH, temp=0.7, max_tokens=2048)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `agent` | `str` | required | Model name |
| `prompt` | `str` | synthesis prompt | System prompt for aggregation |
| `temp` | `float` | 0.7 | Sampling temperature |
| `max_tokens` | `int` | 2048 | Maximum response length |

**Behavior**: Reduces N responses to 1. Typically the final step.

```python
Aggregate("gpt-5.2-chat-latest")

# Custom prompt
Aggregate("gpt-5.2-chat-latest", prompt="Select the best response and return it verbatim.")
```

---

### Refine

Improve each response individually.

```python
Refine(agents, prompt=P_REFINE, temp=0.7, max_tokens=2048)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `agents` | `list[str]` | required | Model names (cycled if fewer than responses) |
| `prompt` | `str` | refinement prompt | Template with `{text}` and `{query}` placeholders |
| `temp` | `float` | 0.7 | Sampling temperature |
| `max_tokens` | `int` | 2048 | Maximum response length |

**Behavior**: Each response is refined independently. Agents are cycled if there are more responses than agents.

```python
# Use GPT-5.2 to refine all responses
Refine(["gpt-5.2-chat-latest"])

# Different refiners for each response
Refine(["gpt-5.2-chat-latest", "claude-sonnet-4-5"])
```

---

### Rank

Select the top N responses by quality.

```python
Rank(agent, n=3, prompt=P_RANK, temp=0.7, max_tokens=2048)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `agent` | `str` | required | Model to perform ranking |
| `n` | `int` | 3 | Number of responses to keep |
| `prompt` | `str` | ranking prompt | Template with `{query}`, `{responses}`, `{n}` placeholders |
| `temp` | `float` | 0.7 | Sampling temperature |
| `max_tokens` | `int` | 2048 | Maximum response length |

**Behavior**: LLM returns comma-separated indices of best responses. Falls back to first N if parsing fails.

```python
Rank("gpt-5.2-chat-latest", n=3)
```

---

### Vote

Identify consensus or select the most accurate answer.

```python
Vote(agent, prompt=P_VOTE, temp=0.7, max_tokens=2048)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `agent` | `str` | required | Model to perform voting |
| `prompt` | `str` | voting prompt | System prompt for consensus finding |
| `temp` | `float` | 0.7 | Sampling temperature |
| `max_tokens` | `int` | 2048 | Maximum response length |

**Behavior**: Reduces N responses to 1 by finding consensus or selecting the best.

```python
Vote("gpt-5.2-chat-latest")
```

---

## Transform Steps

These steps manipulate responses without making LLM calls.

### Shuffle

Randomize response order to prevent positional bias.

```python
Shuffle()
```

**Behavior**: Randomly reorders responses. Some models exhibit position bias (favoring first or last responses).

---

### Dropout

Randomly drop responses with a given probability.

```python
Dropout(rate)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `rate` | `float` | Probability of dropping each response (0.0–1.0) |

**Behavior**: Each response is independently dropped with probability `rate`. If all responses would be dropped, one is kept randomly.

```python
Dropout(0.2)  # 20% chance to drop each response
```

---

### Sample

Take a random subset of responses.

```python
Sample(n)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `n` | `int` | Number of responses to sample |

**Behavior**: Randomly selects `n` responses. If fewer than `n` exist, returns all.

```python
Sample(3)  # Keep 3 random responses
```

---

### Take

Keep the first N responses.

```python
Take(n)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `n` | `int` | Number of responses to keep |

**Behavior**: Deterministic—always keeps the first `n` responses.

```python
Take(3)  # Keep first 3 responses
```

---

### Filter

Keep responses matching a predicate.

```python
Filter(fn)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `fn` | `Callable[[str], bool]` | Function that returns True for responses to keep |

```python
# Keep only responses mentioning "quantum"
Filter(lambda r: "quantum" in r.lower())

# Keep responses over 100 chars
Filter(lambda r: len(r) > 100)
```

---

### Map

Transform each response.

```python
Map(fn)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `fn` | `Callable[[str], str]` | Function to apply to each response |

```python
# Truncate responses
Map(lambda r: r[:500])

# Strip whitespace
Map(str.strip)
```

---

## Default Prompts

The library uses these default prompts:

**Synthesis prompt** (`P_SYNTH`):
```
You have been provided with responses from various models to a query.
Synthesize into a single, high-quality response.
Critically evaluate—some may be biased or incorrect.
Do not simply replicate; offer a refined, accurate reply.
```

**Refinement prompt** (`P_REFINE`):
```
Improve this response:

{text}

Original query: {query}
```

**Voting prompt** (`P_VOTE`):
```
These responses answer the same question.
Identify the consensus view shared by the majority.
If no clear consensus, select the most accurate answer.
Return only that answer, restated clearly.
```

**Ranking prompt** (`P_RANK`):
```
Rank these responses by quality for the query: '{query}'

{responses}

Return the top {n} as comma-separated numbers (e.g., '3, 1, 5').
```
