"""
Basic MoA example using OpenAI.

Requirements:
    pip install mixture-llm[openai]
    export OPENAI_API_KEY=sk-...
"""

import asyncio

from openai import AsyncOpenAI

from mixture_llm import Aggregate, Propose, run
from utils import print_results

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

    print("Running pipeline...")
    result, history = await run(pipeline, query, openai_client)

    print_results(pipeline, query, result, history)


if __name__ == "__main__":
    asyncio.run(main())
