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


def print_history(history: list[dict]) -> None:
    """Pretty print pipeline execution history."""
    print("\n" + "=" * 60)
    print("PIPELINE EXECUTION HISTORY")
    print("=" * 60)

    total_tokens_in = 0
    total_tokens_out = 0
    total_time = 0

    for i, step in enumerate(history, 1):
        print(f"\n[Step {i}] {step['step']}")
        print(f"  Time: {step['step_time']:.2f}s")
        print(f"  Outputs: {len(step['outputs'])}")

        # LLM call details
        if step["llm_calls"]:
            print("  LLM calls:")
            for call in step["llm_calls"]:
                model = call["model"]
                time = call["time"]
                tokens_in = call["in_tokens"]
                tokens_out = call["out_tokens"]

                total_tokens_in += tokens_in
                total_tokens_out += tokens_out

                if "error" in call:
                    print(f"    ✗ {model}: {call['error']}")
                else:
                    print(f"    ✓ {model}: {tokens_in}→{tokens_out} tokens, {time:.2f}s")

        total_time += step["step_time"]

        # Show output previews
        print("  Output previews:")
        for j, output in enumerate(step["outputs"][:3], 1):
            preview = output[:80].replace("\n", " ")
            print(f"    {j}. {preview}...")
        if len(step["outputs"]) > 3:
            print(f"    ... and {len(step['outputs']) - 3} more")

    print("\n" + "-" * 60)
    print("SUMMARY")
    print("-" * 60)
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Total tokens: {total_tokens_in:,} in, {total_tokens_out:,} out")
    print(f"  Total LLM calls: {sum(len(s['llm_calls']) for s in history)}")


async def main():
    pipeline = [
        Propose(["llama-3.3-70b-versatile"] * 4, temp=0.7, max_tokens=256),
        Shuffle(),
        Rank("llama-3.3-70b-versatile", n=2),
        Aggregate("llama-3.3-70b-versatile", max_tokens=512),
    ]

    query = "What makes a good API design?"

    print(f"Query: {query}")
    print("Pipeline: Propose(4) → Shuffle → Rank(top 2) → Aggregate")

    result, history = await run(pipeline, query, groq_client)

    print_history(history)

    print("\n" + "=" * 60)
    print("FINAL RESULT")
    print("=" * 60)
    print(result)

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
