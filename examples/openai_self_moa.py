"""
Self-MoA: Single model sampled multiple times.

From Li et al. (2025) - "Rethinking Mixture-of-Agents"
Self-MoA outperforms standard MoA by 6.6% on AlpacaEval 2.0.

Requirements:
    pip install mixture-llm[openai]
    export OPENAI_API_KEY=sk-...
"""

import asyncio

from openai import AsyncOpenAI

from mixture_llm import Aggregate, Propose, run

client = AsyncOpenAI()


async def openai_client(model, messages, temp, max_tokens):
    # GPT-5.2 requires max_completion_tokens instead of max_tokens
    token_param = (
        {"max_completion_tokens": max_tokens}
        if model.startswith("gpt-5")
        else {"max_tokens": max_tokens}
    )
    resp = await client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temp,
        **token_param,
    )
    return (
        resp.choices[0].message.content,
        resp.usage.prompt_tokens,
        resp.usage.completion_tokens,
    )


async def main():
    # Self-MoA: same model, 6 samples, temperature 0.7 for diversity
    # This configuration can outperform diverse model mixtures
    pipeline = [
        Propose(["gpt-5.2-chat-latest"] * 6, temp=0.7, max_tokens=512),
        Aggregate("gpt-5.2-chat-latest", max_tokens=1024),
    ]

    query = "Explain the implications of quantum computing for cryptography"

    print(f"Query: {query}\n")
    print("Running Self-MoA (6 samples from gpt-5.2-chat-latest)...")

    result, history = await run(pipeline, query, openai_client)

    print(f"\n{'=' * 60}\n")
    print(result)

    # Show individual proposals
    print(f"\n{'=' * 60}")
    print("Individual proposals (first 200 chars each):\n")
    for i, output in enumerate(history[0]["outputs"], 1):
        preview = output[:200].replace("\n", " ")
        print(f"  {i}. {preview}...")

    # Token usage
    total_in = sum(c["in_tokens"] for h in history for c in h["llm_calls"])
    total_out = sum(c["out_tokens"] for h in history for c in h["llm_calls"])
    total_time = sum(h["step_time"] for h in history)
    print(f"\n{'=' * 60}")
    print(f"Tokens: {total_in:,} in, {total_out:,} out")
    print(f"Total time: {total_time:.2f}s")


if __name__ == "__main__":
    asyncio.run(main())
