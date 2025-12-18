"""
Inspecting pipeline execution history.

The `run` function returns detailed history of each step,
useful for debugging, analysis, and cost tracking.

Requirements:
    pip install mixture-llm[litellm]
    export GROQ_API_KEY=gsk_...
"""

import asyncio
import json

from litellm import acompletion

from mixture_llm import Aggregate, Propose, Rank, Shuffle, run


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


async def main():
    pipeline = [
        Propose(["llama-3.3-70b-versatile"] * 4, temp=0.7, max_tokens=256),
        Shuffle(),
        Rank("llama-3.3-70b-versatile", n=2),
        Aggregate("llama-3.3-70b-versatile", max_tokens=512),
    ]

    query = "What makes a good API design?"

    # Show pipeline configuration
    print("Pipeline:")
    print("  Step 1: Propose")
    print("    Models: llama-3.3-70b-versatile (x4)")
    print("  Step 2: Shuffle")
    print("  Step 3: Rank")
    print("    Model: llama-3.3-70b-versatile")
    print("    Keep: top 2")
    print("  Step 4: Aggregate")
    print("    Model: llama-3.3-70b-versatile")
    print()
    print(f"Query: {query}")
    print()
    print("Running pipeline...")

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

    # Save full history to JSON for analysis
    history_json = []
    for step in history:
        history_json.append(
            {
                "step": step["step"],
                "step_time": step["step_time"],
                "num_outputs": len(step["outputs"]),
                "llm_calls": step["llm_calls"],
            }
        )

    with open("history.json", "w") as f:
        json.dump(history_json, f, indent=2)
    print("\nFull history saved to history.json")


if __name__ == "__main__":
    asyncio.run(main())
