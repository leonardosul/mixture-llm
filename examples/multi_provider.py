"""
Multi-provider MoA: Mix OpenAI and Anthropic models.

Uses OpenAI SDK for both providers (Anthropic supports OpenAI-compatible API).

Requirements:
    pip install mixture-llm[openai]
    export OPENAI_API_KEY=sk-...
    export ANTHROPIC_API_KEY=sk-ant-...
"""

import asyncio
import os

from openai import AsyncOpenAI

from mixture_llm import Aggregate, Propose, Shuffle, run

# Two clients, same SDK
openai_client = AsyncOpenAI()
anthropic_client = AsyncOpenAI(
    base_url="https://api.anthropic.com/v1/",
    api_key=os.environ["ANTHROPIC_API_KEY"],
)


async def multi_provider_client(model, messages, temp, max_tokens):
    """Route to appropriate provider based on model name."""
    client = anthropic_client if model.startswith("claude") else openai_client

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
    # Mix providers: GPT-4o and Claude as proposers, Claude aggregates
    pipeline = [
        Propose(
            [
                "gpt-4o",
                "claude-sonnet-4-20250514",
                "gpt-4o-mini",
            ],
            temp=0.7,
            max_tokens=512,
        ),
        Shuffle(),  # Prevent position bias
        Aggregate("claude-sonnet-4-20250514", max_tokens=1024),
    ]

    query = "What are the most promising approaches to aligning AI systems with human values?"

    print(f"Query: {query}\n")
    print("Running multi-provider MoA...")
    print("  Proposers: gpt-4o, claude-sonnet-4-20250514, gpt-4o-mini")
    print("  Aggregator: claude-sonnet-4-20250514\n")

    result, history = await run(pipeline, query, multi_provider_client)

    print(f"{'=' * 60}\n")
    print(result)

    # Show which models were called
    print(f"\n{'=' * 60}")
    print("LLM calls:")
    for step in history:
        for call in step["llm_calls"]:
            status = "✓" if "error" not in call else f"✗ {call['error']}"
            print(f"  {call['model']}: {call['time']:.2f}s {status}")

    total_time = sum(h["step_time"] for h in history)
    print(f"\nTotal time: {total_time:.2f}s")


if __name__ == "__main__":
    asyncio.run(main())
