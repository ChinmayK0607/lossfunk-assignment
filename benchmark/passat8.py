#!/usr/bin/env python
"""

====================================


• Computes *pass @ 8* accuracy on CK0607/gsm8k-1000-extra.
• Produces two CSVs – one for the full 1 000-row split, one for the first 100 rows.
• Outputs land in passat8/1000/ and passat8/100/.

Columns (one row per **question**):
    question          — original GSM8K question
    responses         — list of 8 raw model completions (JSON-encoded)
    model_answers     — list of 8 extracted numeric answers
    correct_answer    — ground-truth numeric answer (from '####' field)
    verdict           — “correct” if ANY of the 8 answers matches, else “wrong”

Run on Modal with:

    modal run passat8.py
"""
# ──────────────────────────────────────────────────────────────
# 0. Imports
# ──────────────────────────────────────────────────────────────
import os, io, csv, re, json, modal, pandas as pd
from datasets import load_dataset
from tqdm.auto import tqdm
from typing import Optional

# ──────────────────────────────────────────────────────────────
# 1. Modal app + image
# ──────────────────────────────────────────────────────────────
app  = modal.App("qwen_vllm_benchmark_passat8")
GPU  = "A100"                             # change to "H100" if desired

image = (
    modal.Image.debian_slim()
    .apt_install("git", "curl", "gcc", "libgomp1", "build-essential", "cmake")
    .pip_install(
        "torch",                          # vLLM runtime dep
        "vllm",
        "datasets>=2.19",
        "pandas",
        "tqdm>=4.66",
        "huggingface-hub"
    )
)

# ──────────────────────────────────────────────────────────────
# 2. Helper regexes & utils
# ──────────────────────────────────────────────────────────────
_ANS_PATTERNS = [
    re.compile(r"<answer>\s*([\s\S]*?)\s*</answer>", re.I),            # <answer> … </answer>
    re.compile(r"</answer>\s*(?:\$|USD\s*)?(-?\d+(?:\.\d+)?)", re.I),  # </answer> $123
    re.compile(r"\\boxed\{([^{}]+)\}"),                                # \boxed{123}
]
_FALLBACK_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")
_STRIP_NON_NUM   = re.compile(r"[^0-9.\-]")

def _extract_model_answer(text: str) -> Optional[str]:
    for pat in _ANS_PATTERNS:
        if (m := pat.search(text)):
            return m.group(1).strip()
    nums = _FALLBACK_NUMBER.findall(text)
    return nums[-1] if nums else None

def _extract_hash_answer(text: str) -> Optional[str]:
    return text.split("####")[1].strip() if "####" in text else None

def _normalize(num: str | None) -> str | None:
    if num is None: return None
    return _STRIP_NON_NUM.sub("", num).lstrip("0") or "0"

# ──────────────────────────────────────────────────────────────
# 3. Benchmark function (remote)
# ──────────────────────────────────────────────────────────────
@app.function(image=image, gpu=GPU, timeout=2 * 60 * 60)   # 2-hour cap
def run_benchmark(num_samples: int) -> bytes:
    """
    Evaluate on `num_samples` rows (1 000 or 100).
    Returns CSV bytes (to be written locally by the entrypoint).
    """
    # 3.1  Dependencies
    import json, pandas as pd
    from vllm import LLM
    from vllm.sampling_params import SamplingParams
    from huggingface_hub import login

    # 3.2  HF auth (token kept exactly as requested)
    login(token="")

    # 3.3  Initialise vLLM engine
    llm = LLM(
        model               = "CK0607/llama3.1-8b-perplexity-rewards-250",
        dtype               = "auto",
        tokenizer_mode      = "auto",
        trust_remote_code   = True,
        tensor_parallel_size= 1,
    )

    # 3.4  Sampling params – we want 8 completions per prompt (pass@8)
    sampling = SamplingParams(
        temperature = 0.6,
        top_p       = 0.95,
        max_tokens  = 512,
        n           = 8,        # ← key change
    )

    # 3.5  Load dataset deterministically
    ds = load_dataset("CK0607/gsm8k-1000-extra", split="train")
    if num_samples < len(ds):
        ds = ds.shuffle(seed=42).select(range(num_samples))

    # 3.6  Build prompts
    sys_prompt = (
        "Reason step by step and enclose your answer strictly "
        "in <answer> </answer> tags.\n\n"
    )
    prompts = [f"{sys_prompt}{q}\nAnswer:\n" for q in ds["question"]]

    # 3.7  Run generation
    outputs = llm.generate(prompts, sampling)

    # 3.8  Evaluate pass@8
    rows = []
    for prompt, sample, out in zip(prompts, ds, outputs):
        # Collect 8 raw completions (strip prompt echo if present)
        completions = []
        answers     = []
        for o in out.outputs:
            txt = o.text[len(prompt):] if o.text.startswith(prompt) else o.text
            completions.append(txt.strip())
            answers.append(_normalize(_extract_model_answer(txt)))

        correct_ans = _normalize(_extract_hash_answer(sample["answer"]))
        verdict     = "correct" if correct_ans in answers else "wrong"

        rows.append({
            "question"       : sample["question"],
            "responses"      : json.dumps(completions, ensure_ascii=False),
            "model_answers"  : json.dumps(answers, ensure_ascii=False),
            "correct_answer" : correct_ans,
            "verdict"        : verdict,
        })

    # 3.9  Return CSV as bytes
    buf = io.StringIO()
    pd.DataFrame(rows).to_csv(buf, index=False)
    return buf.getvalue().encode()

# ──────────────────────────────────────────────────────────────
# 4. Local entrypoint
# ──────────────────────────────────────────────────────────────
@app.local_entrypoint()
def main():
    """Runs pass@8 on 1 000 rows and 100-row subset, saves two CSVs."""
    for n in (1000, 100):
        print(f"\n➡️  Running pass@8 benchmark on {n} samples …")
        csv_bytes = run_benchmark.remote(n)

        # Prepare folders & filenames
        out_dir  = os.path.join("passat8", str(n))
        os.makedirs(out_dir, exist_ok=True)
        out_file = os.path.join(out_dir, f"perplexity_gsm8k_passat8_{n}.csv")

        with open(out_file, "wb") as f:
            f.write(csv_bytes)

        print(f"   ✅ Saved → {out_file}")
