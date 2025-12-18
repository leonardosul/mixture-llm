import asyncio
import json
from asyncio import sleep

import evaluate
from datasets import load_dataset
from litellm import acompletion

from mixture_llm import Aggregate, Dropout, Propose, Shuffle, run


# -----------------
# LLM client (LiteLLM -> Groq)
# export GROQ_API_KEY="..."
# -----------------
async def client(model, messages, temp, max_tokens):
    resp = await acompletion(
        model=model, messages=messages, temperature=temp, max_tokens=max_tokens
    )
    text = resp.choices[0].message.content or ""
    return text, resp.usage.prompt_tokens, resp.usage.completion_tokens


# -----------------
# Fair comparison setup
# Baseline uses the SAME model as MoA aggregator
# -----------------
PROPOSERS = [
    "groq/llama-3.3-70b-versatile",
    "groq/moonshotai/kimi-k2-instruct-0905",
    "groq/openai/gpt-oss-20b",
    "groq/qwen/qwen3-32b",
]
AGG = "groq/openai/gpt-oss-120b"

SYSTEM = "You are a faithful summarizer. Do not invent details."
PROMPT_TMPL = "Summarize in ONE sentence:\n\n{text}"

moa_pipe = [
    Propose(PROPOSERS, temp=0.4, max_tokens=128),
    Shuffle(),
    Dropout(0.15),
    Aggregate(
        AGG,
        temp=0.2,
        max_tokens=128,
        prompt="Pick/merge the best candidate. Must be faithful to the source. One sentence.",
    ),
]


async def baseline_summary(text: str) -> str:
    resp = await acompletion(
        model=AGG,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": PROMPT_TMPL.format(text=text)},
        ],
        temperature=0.2,
        max_tokens=128,
    )
    return resp.choices[0].message.content or ""


async def moa_summary(text: str) -> str:
    prompt = PROMPT_TMPL.format(text=text)
    out, _history = await run(moa_pipe, prompt, client)
    return out


async def main(n=10):
    ds = load_dataset("sentence-transformers/xsum", split=f"train[:{n}]")
    refs, preds_base, preds_moa = [], [], []

    for ex in ds:
        await sleep(5)
        doc, ref = ex["article"], ex["summary"]
        refs.append(ref)
        preds_base.append(await baseline_summary(doc))
        preds_moa.append(await moa_summary(doc))

    rouge = evaluate.load("rouge")
    bert = evaluate.load("bertscore")

    r_base = rouge.compute(predictions=preds_base, references=refs)
    r_moa = rouge.compute(predictions=preds_moa, references=refs)

    b_base = bert.compute(predictions=preds_base, references=refs, lang="en")
    b_moa = bert.compute(predictions=preds_moa, references=refs, lang="en")

    def avg(xs):
        return sum(xs) / len(xs)

    print("ROUGE (baseline):", {k: round(v, 4) for k, v in r_base.items()})
    print("ROUGE (MoA):     ", {k: round(v, 4) for k, v in r_moa.items()})
    print("BERTScore-F1 (baseline):", round(avg(b_base["f1"]), 4))
    print("BERTScore-F1 (MoA):     ", round(avg(b_moa["f1"]), 4))

    # optional: save samples for qualitative inspection
    with open("samples.jsonl", "w", encoding="utf-8") as f:
        for i in range(n):
            f.write(
                json.dumps(
                    {
                        "ref": refs[i],
                        "baseline": preds_base[i],
                        "moa": preds_moa[i],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


if __name__ == "__main__":
    asyncio.run(main())
