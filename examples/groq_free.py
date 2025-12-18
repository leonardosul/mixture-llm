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

    # Show pipeline configuration
    print("Pipeline:")
    print("  Step 1: Propose")
    print(f"    Models: {', '.join(GROQ_FREE)}")
    print("  Step 2: Shuffle")
    print("  Step 3: Dropout (20%)")
    print("  Step 4: Aggregate")
    print("    Model: llama-3.3-70b-versatile")
    print()
    print(f"Query: {query}")
    print()
    print("Running MoA with Groq free tier...")

    result, history = await run(pipeline, query, groq_client)

    # Show final output
    print(f"\n{'=' * 60}")
    print("OUTPUT:")
    print(f"{'=' * 60}\n")
    print(result)

    # Show individual proposals (first 100 chars each)
    print(f"\n{'=' * 60}")
    print("PROPOSALS (first 100 chars each):")
    print(f"{'=' * 60}")
    for i, output in enumerate(history[0]["outputs"], 1):
        preview = output[:100].replace("\n", " ")
        print(f"  {i}. {preview}...")

    # Show LLM calls with details
    print(f"\n{'=' * 60}")
    print("LLM CALLS:")
    print(f"{'=' * 60}")
    for step in history:
        if step["llm_calls"]:
            print(f"\n  {step['step']}:")
            for call in step["llm_calls"]:
                status = "✓" if "error" not in call else f"✗ {call['error']}"
                tokens = f"{call['in_tokens']:,} in / {call['out_tokens']:,} out"
                print(f"    {call['model']}: {call['time']:.2f}s | {tokens} | {status}")

    # Show totals
    total_in = sum(c["in_tokens"] for h in history for c in h["llm_calls"])
    total_out = sum(c["out_tokens"] for h in history for c in h["llm_calls"])
    total_time = sum(h["step_time"] for h in history)
    print(f"\n{'=' * 60}")
    print("TOTALS:")
    print(f"{'=' * 60}")
    print(f"  Time: {total_time:.2f}s")
    print(f"  Tokens: {total_in:,} in / {total_out:,} out")


async def self_moa_example():
    """Self-MoA variant using single Groq model."""
    print("\n" + "=" * 60)
    print("SELF-MOA EXAMPLE")
    print("=" * 60)

    pipeline = [
        Propose(["llama-3.3-70b-versatile"] * 4, temp=0.7, max_tokens=512),
        Aggregate("llama-3.3-70b-versatile", max_tokens=1024),
    ]

    query = "Explain the concept of technical debt in software engineering"

    # Show pipeline configuration
    print("\nPipeline:")
    print("  Step 1: Propose")
    print("    Models: llama-3.3-70b-versatile (x4)")
    print("  Step 2: Aggregate")
    print("    Model: llama-3.3-70b-versatile")
    print()
    print(f"Query: {query}")
    print()
    print("Running Self-MoA with Groq...")

    result, history = await run(pipeline, query, groq_client)

    # Show final output
    print(f"\n{'=' * 60}")
    print("OUTPUT:")
    print(f"{'=' * 60}\n")
    print(result)

    # Show individual proposals (first 100 chars each)
    print(f"\n{'=' * 60}")
    print("PROPOSALS (first 100 chars each):")
    print(f"{'=' * 60}")
    for i, output in enumerate(history[0]["outputs"], 1):
        preview = output[:100].replace("\n", " ")
        print(f"  {i}. {preview}...")

    # Show LLM calls with details
    print(f"\n{'=' * 60}")
    print("LLM CALLS:")
    print(f"{'=' * 60}")
    for step in history:
        if step["llm_calls"]:
            print(f"\n  {step['step']}:")
            for call in step["llm_calls"]:
                status = "✓" if "error" not in call else f"✗ {call['error']}"
                tokens = f"{call['in_tokens']:,} in / {call['out_tokens']:,} out"
                print(f"    {call['model']}: {call['time']:.2f}s | {tokens} | {status}")

    # Show totals
    total_in = sum(c["in_tokens"] for h in history for c in h["llm_calls"])
    total_out = sum(c["out_tokens"] for h in history for c in h["llm_calls"])
    total_time = sum(h["step_time"] for h in history)
    print(f"\n{'=' * 60}")
    print("TOTALS:")
    print(f"{'=' * 60}")
    print(f"  Time: {total_time:.2f}s")
    print(f"  Tokens: {total_in:,} in / {total_out:,} out")


async def run_all():
    await main()
    await self_moa_example()


if __name__ == "__main__":
    asyncio.run(run_all())
