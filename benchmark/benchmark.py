#!/usr/bin/env python
"""
"""

# ──────────────────────────────────────────────────────────────
# 0. Imports
# ──────────────────────────────────────────────────────────────
import io, csv, re, modal, pandas as pd
from datasets import load_dataset
from tqdm.auto import tqdm
from typing import Optional

# ──────────────────────────────────────────────────────────────
# 1. Modal app + image
# ──────────────────────────────────────────────────────────────
app = modal.App("qwen_vllm_benchmark")

GPU  = "A100"                     # switch to "A100"/"H100" if desired

image = (
    modal.Image.debian_slim()
    .apt_install("git", "curl", "gcc", "libgomp1", "build-essential", "cmake")
    .pip_install(
        "torch",                  # vLLM’s runtime dep
        "vllm",
        "datasets>=2.19",
        "pandas",
        "tqdm>=4.66",
    )
)

# app.add_secret(modal.Secret.from_name("huggingface-token"))

# ──────────────────────────────────────────────────────────────
# 2. Helper functions
# ──────────────────────────────────────────────────────────────
_ANS_TAG_RE   = re.compile(r"<answer>(.*?)</answer>", re.I | re.S)
_BOXED_RE     = re.compile(r"\\boxed\{([^{}]+)\}")
_NUMBER_RE    = re.compile(r"-?\d+(?:\.\d+)?")
_STRIP_RE     = re.compile(r"[^0-9\.\-]")

def extract_hash_answer(text: str) -> Optional[str]:
    """Numeric answer after '####' in GSM8K labels."""
    return text.split("####")[1].strip() if "####" in text else None

import re
from typing import Optional

_ANS_PATTERNS = [
    # 1️⃣  <answer> … </answer>
    re.compile(r"<answer>\s*([\s\S]*?)\s*</answer>", re.I),
    # 2️⃣  </answer>  $585   (number right after the closing tag, optional $/currency)
    re.compile(r"</answer>\s*(?:\$|USD\s*)?(-?\d+(?:\.\d+)?)", re.I),
    # 3️⃣  \boxed{…}
    re.compile(r"\\boxed\{([^{}]+)\}"),
]

_FALLBACK_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")

def extract_model_answer(text: str) -> Optional[str]:
    """
    Grab the numeric answer from model output, handling patterns like
    • “… <answer> 585 </answer>”
    • “… </answer> $585”
    • “… \\boxed{585}”
    • otherwise fallback to last standalone number.
    """
    for pat in _ANS_PATTERNS:
        m = pat.search(text)
        if m:
            return m.group(1).strip()
    nums = _FALLBACK_NUMBER.findall(text)
    return nums[-1] if nums else None


def normalize(num_txt: str | None) -> str | None:
    if num_txt is None:
        return None
    return _STRIP_RE.sub("", num_txt).lstrip("0") or "0"

# ──────────────────────────────────────────────────────────────
# 3. Remote bench job
# ──────────────────────────────────────────────────────────────
@app.function(image=image, gpu=GPU, timeout=2 * 60 * 60)  # 2-hour cap
def run_benchmark() -> bytes:
    import os
    from vllm import LLM
    from vllm.sampling_params import SamplingParams

    

    # 3.1 Initialise vLLM engine
    llm = LLM(
        model             = "CK0607/unsloth-trained-qwen-soft-reasoning", # update model name to benchmark
        dtype             = "auto",
        tokenizer_mode    = "auto",
        trust_remote_code = True,
        tensor_parallel_size = 1,
    )

    sampling = SamplingParams(
        temperature = 0.6,
        top_p       = 0.95,
        max_tokens  = 2048,
    )

    # 3.2 Load and sample dataset deterministically
    ds = (
        load_dataset("CK0607/gsm8k-1000-extra", split="train")
        .shuffle(seed=42)
        .select(range(1000))
    )

    # 3.3 Prepare prompts
    prompts = []
    system_prompt = "Reason step by step and enclose your answer strictly in <answer> </answer> tags.\n\n"
    for q in ds["question"]:
        prompts.append(f"{system_prompt}{q}\nAnswer:\n")

    outputs   = llm.generate(prompts, sampling)
    rows      = []
    for prompt, item, out in zip(prompts, ds, outputs):
        resp_full   = out.outputs[0].text
        # remove prompt echo if present
        response    = resp_full[len(prompt):] if resp_full.startswith(prompt) else resp_full
        model_ans   = normalize(extract_model_answer(response))
        correct_ans = normalize(extract_hash_answer(item["answer"]))

        rows.append({
            "question":        item["question"],
            "response":        response.strip(),
            "model_answer":    model_ans if model_ans is not None else "null",
            "correct_answer":  correct_ans,
            "verdict":         "correct" if model_ans == correct_ans else "wrong",
        })

    # 3.4 Save CSV → bytes
    df  = pd.DataFrame(rows)
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode()

# ──────────────────────────────────────────────────────────────
# 4. Local entrypoint
# ──────────────────────────────────────────────────────────────
@app.local_entrypoint()
def main():
    csv_bytes = run_benchmark.remote()
    out_file  = "qwen_gsm8k_100_results_soft_rewards.csv" # update file name to benchmark
    with open(out_file, "wb") as f:
        f.write(csv_bytes)
    print(f"\nSaved CSV with 100-sample results → {out_file}")
