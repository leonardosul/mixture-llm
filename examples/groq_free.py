"""
MoA with Groq's free tier models.

Groq offers free access to several models with fast inference.
Great for experimentation without API costs.

Requirements:
    pip install mixture-llm[litellm]
    export GROQ_API_KEY=gsk_...

Note: Free tier has rate limits (30 RPM for most models).
See https://console.groq.com/docs/rate-limits
"""

import asyncio

from litellm import acompletion

from mixture_llm import Aggregate, Dropout, Propose, Shuffle, run
from utils import print_results


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


# Free Groq models (check docs for current list)
GROQ_FREE = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "qwen/qwen3-32b",
    "meta-llama/llama-4-scout-17b-16e-instruct",
]


async def main():
    # Diverse free models with robustness
    pipeline = [
        Propose(GROQ_FREE, temp=0.7, max_tokens=512),
        Shuffle(),
        Dropout(0.2),
        Aggregate("llama-3.3-70b-versatile", max_tokens=1024),
    ]

    query = "What are the most effective strategies for learning a new programming language?"

    print("Running MoA with Groq free tier...")
    result, history = await run(pipeline, query, groq_client)

    print_results(pipeline, query, result, history)


async def self_moa_example():
    """Self-MoA variant using single Groq model."""
    print("\n" + "=" * 60)
    print("SELF-MOA EXAMPLE")
    print("=" * 60 + "\n")

    pipeline = [
        Propose(["llama-3.3-70b-versatile"] * 4, temp=0.7, max_tokens=512),
        Aggregate("llama-3.3-70b-versatile", max_tokens=1024),
    ]

    query = "Explain the concept of technical debt in software engineering"

    print("Running Self-MoA with Groq...")
    result, history = await run(pipeline, query, groq_client)

    print_results(pipeline, query, result, history)


async def run_all():
    await main()
    await self_moa_example()


if __name__ == "__main__":
    asyncio.run(run_all())
