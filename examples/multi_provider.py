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
    # Mix providers: GPT-5 Nano and Claude as proposers, Claude aggregates
    pipeline = [
        Propose(
            [
                "gpt-5-nano-2025-08-07",
                "claude-sonnet-4-5",
                "gpt-5-nano-2025-08-07",
            ],
            temp=0.7,
            max_tokens=512,
        ),
        Shuffle(),  # Prevent position bias
        Aggregate("claude-sonnet-4-5", max_tokens=1024),
    ]

    query = "What are the most promising approaches to aligning AI systems with human values?"

    print(f"Query: {query}\n")
    print("Running multi-provider MoA...")
    print("  Proposers: gpt-5-nano-2025-08-07, claude-sonnet-4-5, gpt-5-nano-2025-08-07")
    print("  Aggregator: claude-sonnet-4-5\n")

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
