#!/usr/bin/env python
"""

"""

# ──────────────────────────────────────────────────────────────
# 0. Imports
# ──────────────────────────────────────────────────────────────
import io, re, modal, pandas as pd
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
        "torch",                  # vLLM runtime dep
        "vllm",
        "datasets>=2.19",
        "pandas",
        "tqdm>=4.66",
        "huggingface-hub"
    )
)

# If you keep your HF token in Modal secrets, uncomment:
# app.add_secret(modal.Secret.from_name("huggingface-token"))

# ──────────────────────────────────────────────────────────────
# 2. Helper functions
# ──────────────────────────────────────────────────────────────
# Pattern that grabs the very first entry in ["A"]  (or B/C/D …)
_CORRECT_CHOICE_RE = re.compile(r'^\s*\[\s*"([A-Za-z])"\s*\]')

# Model-answer patterns (letter inside <answer> … </answer>)
_MODEL_PATTERNS = [
    re.compile(r"<answer>\s*([A-Za-z])\s*</answer>", re.I),
    re.compile(r"</answer>\s*([A-Za-z])", re.I),          # sometimes they forget to open
]

def extract_correct_choice(text: str) -> Optional[str]:
    m = _CORRECT_CHOICE_RE.match(text)
    return m.group(1).upper() if m else None

def extract_model_choice(text: str) -> Optional[str]:
    # Try explicit patterns
    for pat in _MODEL_PATTERNS:
        m = pat.search(text)
        if m:
            return m.group(1).upper()
    # Fallback: last standalone capital letter A-D
    caps = re.findall(r"\b([A-D])\b", text.upper())
    return caps[-1] if caps else None

# ──────────────────────────────────────────────────────────────
# 3. Remote bench job
# ──────────────────────────────────────────────────────────────
@app.function(image=image, gpu=GPU, timeout=2 * 60 * 60)  # 2-hour cap
def run_benchmark() -> bytes:
    from vllm import LLM
    from vllm.sampling_params import SamplingParams
    from huggingface_hub import login

    # If you stored the token as a secret, you can omit the inline string.
    login(token="")

    # 3.1 Initialise vLLM engine
    llm = LLM(
        model="mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated",
        dtype="auto",
        tokenizer_mode="auto",
        trust_remote_code=True,
        tensor_parallel_size=1,
    )

    sampling = SamplingParams(
        temperature=0.0,    # deterministic for grading MCQ
        max_tokens=256,
    )

    # 3.2 Load & sample dataset deterministically
    ds = (
        load_dataset("CK0607/jee-math-gsm8k", split="train")
        .shuffle(seed=42)
        .select(range(100))
    )

    # 3.3 Prepare prompts
    sys_prompt = (
        "You are an expert mathematics tutor. "
        "Think step-by-step and provide ONLY your final choice (A/B/C/D) "
        "inside <answer> </answer> tags.\n\n"
    )
    prompts = [f"{sys_prompt}{q}\nAnswer:\n" for q in ds["question"]]

    outputs = llm.generate(prompts, sampling)

    rows = []
    for item, prompt, out in zip(ds, prompts, outputs):
        full_resp = out.outputs[0].text
        # strip echoed prompt, if any
        response  = full_resp[len(prompt):] if full_resp.startswith(prompt) else full_resp

        model_ans   = extract_model_choice(response)
        correct_ans = extract_correct_choice(item["answer"])

        rows.append(
            {
                "question":       item["question"],
                "response":       response.strip(),
                "model_answer":   model_ans or "null",
                "correct_answer": correct_ans or "null",
                "verdict":        "correct" if model_ans == correct_ans else "wrong",
            }
        )

    # 3.4 Save CSV → bytes
    import io, pandas as pd
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
    out_file  = "jee_math_base_results.csv"
    with open(out_file, "wb") as f:
        f.write(csv_bytes)
    print(f"\nSaved CSV with 100-sample results → {out_file}")
