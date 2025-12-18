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
    resp = await client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temp,
        max_tokens=max_tokens,
    )
    return (
        resp.choices[0].message.content,
        resp.usage.prompt_tokens,
        resp.usage.completion_tokens,
    )


async def main():
    # Simple 2-layer MoA with GPT-4.1-mini
    pipeline = [
        Propose(["gpt-4.1-mini"] * 3, temp=0.7),
        Aggregate("gpt-4.1-mini"),
    ]

    query = "What are the key differences between Python and Rust?"

    print(f"Query: {query}\n")
    print("Running pipeline...")

    result, history = await run(pipeline, query, openai_client)

    print(f"\n{'=' * 60}\n")
    print(result)

    # Token usage
    total_in = sum(c["in_tokens"] for h in history for c in h["llm_calls"])
    total_out = sum(c["out_tokens"] for h in history for c in h["llm_calls"])
    total_time = sum(h["step_time"] for h in history)
    print(f"\n{'=' * 60}")
    print(f"Tokens: {total_in:,} in, {total_out:,} out")
    print(f"Total time: {total_time:.2f}s")


if __name__ == "__main__":
    asyncio.run(main())
