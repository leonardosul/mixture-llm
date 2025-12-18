"""
Utility functions for mixture-llm examples.
"""


def print_pipeline(pipeline: list) -> None:
    """Print pipeline configuration by inspecting step objects."""
    print("Pipeline:")
    for i, step in enumerate(pipeline, 1):
        step_name = step.__class__.__name__
        print(f"  Step {i}: {step_name}")

        # Extract models/agents from step attributes
        if hasattr(step, "agents"):
            agents = step.agents
            # Check if all agents are the same (Self-MoA pattern)
            if len(set(agents)) == 1 and len(agents) > 1:
                print(f"    Models: {agents[0]} (x{len(agents)})")
            else:
                print(f"    Models: {', '.join(agents)}")
        elif hasattr(step, "agent"):
            print(f"    Model: {step.agent}")

        # Print additional parameters if present
        if hasattr(step, "n") and step_name == "Rank":
            print(f"    Keep: top {step.n}")
        if hasattr(step, "rate") and step_name == "Dropout":
            print(f"    Rate: {int(step.rate * 100)}%")


def print_results(pipeline: list, query: str, result: str, history: list) -> None:
    """Print full results from a pipeline run."""
    # Print pipeline configuration
    print_pipeline(pipeline)
    print()
    print(f"Query: {query}")
    print()

    # Show final output
    print(f"{'=' * 60}")
    print("OUTPUT:")
    print(f"{'=' * 60}\n")
    print(result)

    # Show individual proposals (first 100 chars each)
    if history and history[0]["outputs"]:
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
