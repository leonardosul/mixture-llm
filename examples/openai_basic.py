"""
Basic MoA example using OpenAI.

Requirements:
    pip install mixture-llm[openai]
    export OPENAI_API_KEY=sk-...
"""

import asyncio

from openai import AsyncOpenAI

from mixture_llm import Aggregate, Propose, run

client = AsyncOpenAI()


async def openai_client(model, messages, temp, max_tokens):
    # GPT-5 models require max_completion_tokens, don't support custom temperature, and need reasoning_effort
    is_gpt5 = model.startswith("gpt-5")
    params = {
        "model": model,
        "messages": messages,
        **({"max_completion_tokens": max_tokens, "reasoning_effort": "minimal"} if is_gpt5 else {"max_tokens": max_tokens, "temperature": temp}),
    }
    resp = await client.chat.completions.create(**params)
    return (
        resp.choices[0].message.content,
        resp.usage.prompt_tokens,
        resp.usage.completion_tokens,
    )


async def main():
    # Simple 2-layer MoA with GPT-5 Nano
    pipeline = [
        Propose(["gpt-5-nano-2025-08-07"] * 3, temp=0.7),
        Aggregate("gpt-5-nano-2025-08-07"),
    ]

    query = "What are the key differences between Python and Rust?"

    # Show pipeline configuration
    print("Pipeline:")
    print("  Step 1: Propose")
    print("    Models: gpt-5-nano-2025-08-07 (x3)")
    print("  Step 2: Aggregate")
    print("    Model: gpt-5-nano-2025-08-07")
    print()
    print(f"Query: {query}")
    print()
    print("Running pipeline...")

    result, history = await run(pipeline, query, openai_client)

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


if __name__ == "__main__":
    asyncio.run(main())
