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

    # Show pipeline configuration
    print("Pipeline:")
    print("  Step 1: Propose")
    print(f"    Models: {', '.join(PROPOSERS)}")
    print("  Step 2: Synthesize")
    print(f"    Models: {', '.join(PROPOSERS)}")
    print("  Step 3: Aggregate")
    print("    Model: qwen/qwen-2.5-72b-instruct")
    print()
    print(f"Query: {query}")
    print()
    print("Running 3-layer MoA via OpenRouter...")

    result, history = await run(pipeline, query, openrouter_client)

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


if __name__ == "__main__":
    asyncio.run(main())
