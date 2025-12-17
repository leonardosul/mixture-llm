import pytest

from mixture_llm import Aggregate, Propose, Shuffle, Take, run


async def mock_client(model, messages, temp, max_tokens):
    return f"Response from {model}", 10, 10


@pytest.mark.asyncio
async def test_propose_aggregate():
    pipeline = [Propose(["m1", "m2"]), Aggregate("agg")]
    result, history = await run(pipeline, "test", mock_client)
    assert "agg" in result
    assert len(history) == 2


@pytest.mark.asyncio
async def test_transforms():
    pipeline = [Propose(["m1", "m2", "m3"]), Shuffle(), Take(2)]
    _, history = await run(pipeline, "test", mock_client)
    assert len(history[-1]["outputs"]) == 2
