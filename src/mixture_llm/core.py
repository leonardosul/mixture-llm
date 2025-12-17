import asyncio
import random
import re
import time
from collections.abc import Awaitable, Callable
from itertools import cycle
from typing import Any, NamedTuple, Protocol, TypedDict


class Message(TypedDict):
    role: str
    content: str


class Client(Protocol):
    def __call__(
        self,
        *,
        model: str,
        messages: list[Message],
        temp: float,
        max_tokens: int,
    ) -> Awaitable[tuple[str, int, int]]: ...


DEFAULT_TEMP = 0.7
DEFAULT_MAX_TOKENS = 2048
P_SYNTH = (
    "You have been provided with responses from various models to a query. "
    "Synthesize into a single, high-quality response. "
    "Critically evaluate—some may be biased or incorrect. "
    "Do not simply replicate; offer a refined, accurate reply."
)

P_REFINE = "Improve this response:\n\n{text}\n\nOriginal query: {query}"

P_VOTE = (
    "These responses answer the same question. "
    "Identify the consensus view shared by the majority. "
    "If no clear consensus, select the most accurate answer. "
    "Return only that answer, restated clearly."
)

P_RANK = (
    "Rank these responses by quality for the query: '{query}'\n\n"
    "{responses}\n\n"
    "Return the top {n} as comma-separated numbers (e.g., '3, 1, 5')."
)


class Propose(NamedTuple):
    agents: list[str]
    temp: float = DEFAULT_TEMP
    max_tokens: int = DEFAULT_MAX_TOKENS


class Synthesize(NamedTuple):
    agents: list[str]
    prompt: str = P_SYNTH
    temp: float = DEFAULT_TEMP
    max_tokens: int = DEFAULT_MAX_TOKENS


class Aggregate(NamedTuple):
    agent: str
    prompt: str = P_SYNTH
    temp: float = DEFAULT_TEMP
    max_tokens: int = DEFAULT_MAX_TOKENS


class Refine(NamedTuple):
    agents: list[str]
    prompt: str = P_REFINE
    temp: float = DEFAULT_TEMP
    max_tokens: int = DEFAULT_MAX_TOKENS


class Rank(NamedTuple):
    agent: str
    n: int = 3
    prompt: str = P_RANK
    temp: float = DEFAULT_TEMP
    max_tokens: int = DEFAULT_MAX_TOKENS


class Vote(NamedTuple):
    agent: str
    prompt: str = P_VOTE
    temp: float = DEFAULT_TEMP
    max_tokens: int = DEFAULT_MAX_TOKENS


class Shuffle(NamedTuple): ...


class Dropout(NamedTuple):
    rate: float


class Sample(NamedTuple):
    n: int


class Take(NamedTuple):
    n: int


class Filter(NamedTuple):
    fn: Callable[[str], bool]


class Map(NamedTuple):
    fn: Callable[[str], str]


def _enumerate(responses: list[str]) -> str:
    return "\n\n".join(f"{i + 1}. {x}" for i, x in enumerate(responses))


def _msgs(prompt: str, outs: list[str], query: str) -> list[Message]:
    return [
        {"role": "system", "content": prompt},
        {"role": "user", "content": f"Responses:\n{_enumerate(outs)}\n\nQuery: {query}"},
    ]


async def _call(
    model: str, messages: list[Message], temp: float, max_tokens: int, client: Client
) -> tuple[str | None, dict[str, Any]]:
    t0 = time.time()
    try:
        text, in_tok, out_tok = await client(
            model=model, messages=messages, temp=temp, max_tokens=max_tokens
        )
        return text, {
            "model": model,
            "time": time.time() - t0,
            "in_tokens": in_tok,
            "out_tokens": out_tok,
        }
    except Exception as e:
        return None, {
            "model": model,
            "time": time.time() - t0,
            "in_tokens": 0,
            "out_tokens": 0,
            "error": repr(e),
        }


async def _many(
    models: list[str], messages: list[Message], temp: float, max_tokens: int, client: Client
) -> tuple[list[str], list[dict[str, Any]]]:
    res = await asyncio.gather(*(_call(m, messages, temp, max_tokens, client) for m in models))
    return [t for (t, _) in res if t], [info for (_, info) in res]


def _rank(text: str, *, max_len: int, n: int) -> list[int]:
    out: list[int] = []
    for s in re.findall(r"\d+", text):
        i = int(s) - 1
        if 0 <= i < max_len and i not in out:
            out.append(i)
        if len(out) >= n:
            break
    return out


# TODO: pipeline type annotation
async def run(pipeline: list[Any], query: str, client: Client) -> tuple[str, list[dict[str, Any]]]:
    responses: list[str] = []
    history: list[dict[str, Any]] = []

    for step in pipeline:
        t0 = time.time()
        calls: list[dict[str, Any]] = []

        match step:
            case Propose(agents, temp, max_tokens):
                responses, calls = await _many(
                    agents, [{"role": "user", "content": query}], temp, max_tokens, client
                )

            case Synthesize(agents, prompt, temp, max_tokens):
                if responses:
                    responses, calls = await _many(
                        agents, _msgs(prompt, responses, query), temp, max_tokens, client
                    )

            case Aggregate(agent, prompt, temp, max_tokens):
                if responses:
                    text, info = await _call(
                        agent, _msgs(prompt, responses, query), temp, max_tokens, client
                    )
                    calls = [info]
                    if text:
                        responses = [text]

            case Refine(agents, prompt, temp, max_tokens):
                if responses:
                    msgs: list[list[Message]] = [
                        [{"role": "user", "content": prompt.format(text=o, query=query)}]
                        for o in responses
                    ]
                    res = await asyncio.gather(
                        *(
                            _call(a, m, temp, max_tokens, client)
                            for a, m in zip(cycle(agents), msgs)
                        )
                    )
                    responses, calls = [t for t, _ in res if t], [info for _, info in res]

            case Rank(agent, n, prompt, temp, max_tokens):
                if responses:
                    p = prompt.format(query=query, responses=_enumerate(responses), n=n)
                    text, info = await _call(
                        agent, [{"role": "user", "content": p}], temp, max_tokens, client
                    )
                    calls = [info]
                    if not text:
                        responses = responses[:n]
                    else:
                        idx = _rank(text, max_len=len(responses), n=n)
                        responses = [responses[i] for i in idx] if idx else responses[:n]

            case Vote(agent, prompt, temp, max_tokens):
                if responses:
                    text, info = await _call(
                        agent, _msgs(prompt, responses, query), temp, max_tokens, client
                    )
                    calls = [info]
                    if text:
                        responses = [text]

            case Shuffle():
                if responses:
                    responses = random.sample(responses, len(responses))

            case Dropout(rate):
                prev = responses
                responses = [o for o in responses if random.random() > rate]
                if prev and not responses:
                    responses = [random.choice(prev)]

            case Sample(n):
                responses = random.sample(responses, min(n, len(responses)))

            case Take(n):
                responses = responses[:n]

            case Filter(fn):
                responses = [o for o in responses if fn(o)]

            case Map(fn):
                responses = [fn(o) for o in responses]

        history.append(
            {
                "step": type(step).__name__,
                "outputs": responses.copy(),
                "llm_calls": calls,
                "step_time": time.time() - t0,
            }
        )

    return (responses[0] if responses else ""), history
