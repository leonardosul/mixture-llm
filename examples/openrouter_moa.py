"""
Together MoA configuration via OpenRouter.

Approximates the paper's benchmark-winning setup using OpenRouter
to access multiple model providers through one API.

Requirements:
    pip install mixture-llm[openai]
    export OPENROUTER_API_KEY=sk-or-...
"""

import asyncio
import os

from openai import AsyncOpenAI

from mixture_llm import Aggregate, Propose, Synthesize, run
from utils import print_results

client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)


async def openrouter_client(model, messages, temp, max_tokens):
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


# Modern equivalents of the original Together MoA proposers
PROPOSERS = [
    "qwen/qwen-2.5-72b-instruct",
    "meta-llama/llama-3.3-70b-instruct",
    "mistralai/mixtral-8x22b-instruct",
    "databricks/dbrx-instruct",
]


async def main():
    # 3-layer MoA matching the paper configuration
    # Layer 1: Propose (all models answer independently)
    # Layer 2: Synthesize (all models see all responses)
    # Layer 3: Aggregate (single model produces final answer)
    pipeline = [
        Propose(PROPOSERS, temp=0.7, max_tokens=512),
        Synthesize(PROPOSERS, temp=0.7, max_tokens=512),
        Aggregate("qwen/qwen-2.5-72b-instruct", max_tokens=1024),
    ]

    query = "Compare and contrast the economic policies of keynesianism and monetarism"

    print("Running 3-layer MoA via OpenRouter...")
    result, history = await run(pipeline, query, openrouter_client)

    print_results(pipeline, query, result, history)


if __name__ == "__main__":
    asyncio.run(main())
