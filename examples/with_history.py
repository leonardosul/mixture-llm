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
from utils import print_results


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

    print("Running pipeline...")
    result, history = await run(pipeline, query, groq_client)

    print_results(pipeline, query, result, history)

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
