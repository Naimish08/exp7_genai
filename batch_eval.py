#!/usr/bin/env python3
"""
Batch evaluation for the Generative AI chatbot (EXP 7).

For N random questions from nglif/GenerativeAI-QA-Dataset it stores, in one JSON file:
  - question
  - dataset_reference   : the dataset's own response (often low-quality boilerplate)
  - expected_answer     : a clean reference answer written by the LLM from the dataset response
  - generated_answer    : what the chatbot says (system prompt + domain info + question)
  - timing, model, prompt mode, and a rough "says not available" flag

Setup:
    pip install datasets huggingface_hub
    export HF_TOKEN=hf_...            # Colab: os.environ["HF_TOKEN"] = getpass(...)

Run:
    python batch_eval.py                          # 1000 questions, Prompt B, with expected answers
    python batch_eval.py --n 50                   # quick trial first
    python batch_eval.py --prompt A               # basic prompt, no domain info (Modification 1)
    python batch_eval.py --no-expected            # skip expected-answer generation (half the calls)

It saves progress every few records and resumes if you run it again with the same --out file.
"""

import argparse
import json
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import tomllib

from datasets import load_dataset
from huggingface_hub import InferenceClient

from domain_info import domain_information, extra_facts

DATASET = "nglif/GenerativeAI-QA-Dataset"
DOMAIN_NAME = "Generative AI"
DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"  # use an instructor-approved model you can access

NOT_AVAILABLE = re.compile(
    r"not (available|provided|mentioned|covered|found|specified|included)"
    r"|outside (the|my)|no information|cannot find|can't find|don't have (any )?information"
    r"|does(n't| not) (contain|include|mention|provide)",
    re.IGNORECASE,
)


# ----------------------------------------------------------------------------
# Prompts (same wording as app.py)
# ----------------------------------------------------------------------------
def system_prompt_b(info):
    return f"""You are a helpful domain-specific AI assistant for the domain: {DOMAIN_NAME}.

Use the following domain information when answering the question.

DOMAIN INFORMATION:
{info}

INSTRUCTIONS:
1. Answer clearly and concisely.
2. Use the domain information when relevant.
3. Do not intentionally invent domain-specific facts.
4. If the information is not available, clearly state that the information is not available in the provided information.
5. Do not answer questions outside the domain unless general information is explicitly requested.
"""


SYSTEM_A = "Answer the following question."

EXPECTED_SYSTEM = (
    "You write reference answers for an evaluation set. You are given a question and a "
    "noisy source text. Write a clean, correct, concise answer (2 to 4 sentences) using "
    "ONLY ideas present in the source text. Fix garbled word order and remove filler. "
    "If the source text contains no concrete information that answers the question, "
    "reply exactly: NO_REFERENCE_AVAILABLE"
)


# ----------------------------------------------------------------------------
# LLM helpers
# ----------------------------------------------------------------------------
def chat(client, model, messages, max_tokens, temperature, retries=5):
    """chat_completion with exponential backoff (free tiers rate-limit)."""
    for attempt in range(retries):
        try:
            r = client.chat_completion(
                messages=messages,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return r.choices[0].message.content.strip()
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)


def process_one(client, item, args, system_prompt):
    rec = {
        "id": item["id"],
        "question": item["question"],
        "dataset_reference": item["reference"],
        "expected_answer": None,
        "generated_answer": None,
        "says_not_available": None,
        "prompt_mode": args.prompt,
        "extra_facts": args.extra_facts,
        "model": args.model,
        "latency_s": None,
        "error": None,
    }
    try:
        t0 = time.time()
        answer = chat(
            client,
            args.model,
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": item["question"]},
            ],
            max_tokens=300,
            temperature=args.temperature,
        )
        rec["latency_s"] = round(time.time() - t0, 2)
        rec["generated_answer"] = answer
        rec["says_not_available"] = bool(NOT_AVAILABLE.search(answer))

        if not args.no_expected:
            rec["expected_answer"] = chat(
                client,
                args.model,
                [
                    {"role": "system", "content": EXPECTED_SYSTEM},
                    {
                        "role": "user",
                        "content": f"QUESTION:\n{item['question']}\n\nSOURCE TEXT:\n{item['reference']}",
                    },
                ],
                max_tokens=250,
                temperature=0.0,
            )
    except Exception as e:  # keep going; failed rows are retried on the next run
        rec["error"] = f"{type(e).__name__}: {e}"
    return rec


def load_token():
    token = os.getenv("HF_TOKEN")
    if token:
        return token
    secrets_path = Path(__file__).parent / ".streamlit" / "secrets.toml"
    if secrets_path.exists():
        with secrets_path.open("rb") as secrets_file:
            return tomllib.load(secrets_file).get("HF_TOKEN")
    return None


# ----------------------------------------------------------------------------
# Saving / resuming
# ----------------------------------------------------------------------------
def save(path, meta, records):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "records": records}, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)  # atomic: a crash never leaves a half-written file


def load_existing(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return [r for r in data.get("records", []) if not r.get("error")]


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000, help="number of questions")
    ap.add_argument("--seed", type=int, default=42, help="sampling seed (same seed = same questions)")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--prompt", choices=["A", "B"], default="B", help="A = basic, B = domain-specific")
    ap.add_argument("--extra-facts", action="store_true", help="append the 5 extra facts (Modification 2)")
    ap.add_argument("--no-expected", action="store_true", help="skip LLM expected-answer generation")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--workers", type=int, default=4, help="parallel requests (lower if you get 429s)")
    ap.add_argument("--save-every", type=int, default=10)
    ap.add_argument("--out", default="eval_results.json")
    args = ap.parse_args()

    token = load_token()
    if not token:
        raise SystemExit("Add HF_TOKEN to .streamlit/secrets.toml or set the HF_TOKEN environment variable first.")
    client = InferenceClient(token=token, provider="auto")

    info = domain_information + (extra_facts if args.extra_facts else "")
    system_prompt = SYSTEM_A if args.prompt == "A" else system_prompt_b(info)

    # sample questions
    ds = load_dataset(DATASET, split="train")
    n = min(args.n, len(ds))
    indices = random.Random(args.seed).sample(range(len(ds)), n)
    items = [
        {"id": i, "question": ds[i]["instruction"], "reference": ds[i]["response"]}
        for i in indices
    ]

    # resume
    records = load_existing(args.out)
    done = {r["id"] for r in records}
    todo = [it for it in items if it["id"] not in done]
    print(f"{len(items)} sampled, {len(done)} already done, {len(todo)} to run")

    meta = {
        "dataset": DATASET,
        "n_requested": n,
        "seed": args.seed,
        "model": args.model,
        "prompt_mode": args.prompt,
        "extra_facts": args.extra_facts,
        "expected_answers_generated": not args.no_expected,
        "temperature": args.temperature,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    failed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(process_one, client, it, args, system_prompt) for it in todo]
        for k, fut in enumerate(as_completed(futures), 1):
            rec = fut.result()
            if rec["error"]:
                failed += 1
            else:
                records.append(rec)
            if k % args.save_every == 0:
                save(args.out, meta, records)
                print(f"  {k}/{len(todo)} done ({failed} failed)")
    save(args.out, meta, records)

    # quick summary
    ok = [r for r in records if r["generated_answer"]]
    na = sum(1 for r in ok if r["says_not_available"])
    lat = [r["latency_s"] for r in ok if r["latency_s"] is not None]
    print(f"\nSaved {len(records)} records to {args.out}  ({failed} failed this run; rerun to retry them)")
    if ok:
        print(f"Answers that say 'not available': {na}/{len(ok)} ({100 * na / len(ok):.0f}%)")
        if lat:
            print(f"Mean latency: {sum(lat) / len(lat):.2f}s")


if __name__ == "__main__":
    main()
