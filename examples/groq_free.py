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

    print(f"Query: {query}\n")
    print("Running MoA with Groq free tier...")
    print(f"  Models: {', '.join(GROQ_FREE)}\n")

    result, history = await run(pipeline, query, groq_client)

    print(f"{'=' * 60}\n")
    print(result)

    # Show execution details
    print(f"\n{'=' * 60}")
    print("Execution:")
    total_time = 0
    total_in = 0
    total_out = 0
    for step in history:
        n_outputs = len(step["outputs"])
        n_calls = len(step["llm_calls"])
        print(
            f"  {step['step']}: {n_outputs} outputs, {n_calls} LLM calls, {step['step_time']:.2f}s"
        )
        total_time += step["step_time"]
        total_in += sum(c["in_tokens"] for c in step["llm_calls"])
        total_out += sum(c["out_tokens"] for c in step["llm_calls"])
    print(f"\nTokens: {total_in:,} in, {total_out:,} out")
    print(f"Total time: {total_time:.2f}s")


async def self_moa_example():
    """Self-MoA variant using single Groq model."""
    print("\n" + "=" * 60)
    print("Self-MoA with Groq (single model, 4 samples)")
    print("=" * 60 + "\n")

    pipeline = [
        Propose(["llama-3.3-70b-versatile"] * 4, temp=0.7, max_tokens=512),
        Aggregate("llama-3.3-70b-versatile", max_tokens=1024),
    ]

    query = "Explain the concept of technical debt in software engineering"

    result, history = await run(pipeline, query, groq_client)
    print(result)

    # Show execution details
    print(f"\n{'=' * 60}")
    print("Execution:")
    total_time = 0
    total_in = 0
    total_out = 0
    for step in history:
        n_outputs = len(step["outputs"])
        n_calls = len(step["llm_calls"])
        print(
            f"  {step['step']}: {n_outputs} outputs, {n_calls} LLM calls, {step['step_time']:.2f}s"
        )
        total_time += step["step_time"]
        total_in += sum(c["in_tokens"] for c in step["llm_calls"])
        total_out += sum(c["out_tokens"] for c in step["llm_calls"])
    print(f"\nTokens: {total_in:,} in, {total_out:,} out")
    print(f"Total time: {total_time:.2f}s")


async def run_all():
    await main()
    await self_moa_example()


if __name__ == "__main__":
    asyncio.run(run_all())
