#!/usr/bin/env python
"""
grpo_llama3_hard_sonnets.py - FIXED VERSION
────────────────────────────────────────────────────────────────────────────
Hard-reward GRPO on Shakespeare sonnets + local-artifact download.
"""
import os, re, math, modal, torch, pandas as pd
from datasets import load_dataset
import torch.nn.functional as F

# ─────────────── 0.  CONSTANTS / TOKENS (unchanged) ───────────────
hf_token   = ""
HF_REPO_ID = "CK0607/unsloth-trained-qwen-ce-perplexity"
WANDB_KEY  = ""

GPU_TYPE   = "A100"
MAX_SEQ    = 512
LORA_RANK  = 32
MAX_STEPS  = 50

CSV_NAME   = "hard_sonnets_samples.csv"
TXT_NAME   = "hard_sonnets_samples.txt"

# ─────────────── 1.  MODAL APP / IMAGE / VOLUME ───────────────
stub = modal.App("grpo_llama3_hard_sonnets")

image = (
    modal.Image.debian_slim()
    .apt_install("git", "curl", "gcc", "libgomp1", "build-essential", "cmake")
    .pip_install(
        "unsloth", "trl", "datasets", "tqdm",
        "huggingface_hub", "vllm", "pandas", "pyphen", "wandb"
    )
)

VOLUME = modal.Volume.from_name("unsloth-grpo-sonnet-vol", create_if_missing=True)

# ─────────────── 2.  HARD-REWARD HELPERS ───────────────
import pyphen
_dic = pyphen.Pyphen(lang="en")
_rhyme_pairs = [(0,2),(1,3),(4,6),(5,7),(8,10),(9,11),(12,13)]

def _split_lines(t):       return [ln for ln in re.split(r"\s*\n\s*", t.strip()) if ln]
def _last_word(line):
    m = re.search(r"([A-Za-z']+)[^A-Za-z']*$", line.strip())
    return m.group(1).lower() if m else ""
def _n_syllables(word):    return len(_dic.inserted(word).split("-")) or 1
def _line_syllables(line): return sum(_n_syllables(w) for w in re.findall(r"[A-Za-z']+", line))
def _rhymes(w1, w2):       return w1[-2:] == w2[-2:]    # naïve

# FIXED: Make comps a keyword argument with default None, handle both calling patterns
def hard_linecount_reward(completions=None, comps=None, **kwargs):
    # Handle both possible calling patterns
    target_comps = completions if completions is not None else comps
    if target_comps is None:
        raise ValueError("Either 'completions' or 'comps' must be provided")
    
    return [0.5 if len(_split_lines(c[0]["content"])) == 14 else 0.0 for c in target_comps]

def hard_meter_reward(completions=None, comps=None, **kwargs):
    # Handle both possible calling patterns
    target_comps = completions if completions is not None else comps
    if target_comps is None:
        raise ValueError("Either 'completions' or 'comps' must be provided")
    
    scores=[]
    for c in target_comps:
        lines=_split_lines(c[0]["content"])
        ok = len(lines)==14 and all(8<=_line_syllables(l)<=12 for l in lines)
        scores.append(0.5 if ok else 0.0)
    return scores

def hard_rhyme_reward(completions=None, comps=None, **kwargs):
    # Handle both possible calling patterns
    target_comps = completions if completions is not None else comps
    if target_comps is None:
        raise ValueError("Either 'completions' or 'comps' must be provided")
    
    scores=[]
    for c in target_comps:
        lines=_split_lines(c[0]["content"])
        cond = len(lines)==14 and all(
            _rhymes(_last_word(lines[i]), _last_word(lines[j]))
            for i,j in _rhyme_pairs
        )
        scores.append(0.5 if cond else 0.0)
    return scores

# ─────────────── 3.  REMOTE JOB ───────────────
@stub.function(
    image=image,
    gpu=GPU_TYPE,
    timeout=72000,
    volumes={"/root/artifacts": VOLUME},
)
def run_train_push():
    """
    Finetunes, pushes the merged model, writes CSV & TXT to the Modal
    volume, **returns both files as bytes** so the local entrypoint can
    save them on the caller's machine.
    """
    from unsloth import FastLanguageModel
    from trl import GRPOConfig, GRPOTrainer
    from huggingface_hub import HfApi
    import wandb, torch, pandas as pd

    # ── W&B ──
    wandb.login(key=WANDB_KEY)

    # ── MODEL + LoRA ──
    model, tok = FastLanguageModel.from_pretrained(
        "meta-llama/meta-Llama-3.1-8B-Instruct",
        max_seq_length=MAX_SEQ,
        load_in_4bit=True,
        fast_inference=True,
        max_lora_rank=LORA_RANK,
        gpu_memory_utilization=0.65,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        target_modules=["q_proj","k_proj","v_proj","o_proj",
                        "gate_proj","up_proj","down_proj"],
        lora_alpha=LORA_RANK,
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    # ── DATA ──
    ds = load_dataset("Lambent/shakespeare_sonnets_diffused", split="train")
    SYS_PROMPT = "You are a master poet. Write a Shakespearean sonnet.\n"
    train_ds = ds.map(
        lambda row: {
            "prompt":[
                {"role":"system","content":SYS_PROMPT},
                {"role":"user","content":"Theme: "+row["Variation Text"][:80]},
            ],
            "reference": row["Variation Text"],
        },
        remove_columns=ds.column_names,
    )

    # ── GRPO CONFIG ──
    args = GRPOConfig(
        learning_rate=5e-6, weight_decay=0.1, warmup_ratio=0.06,
        lr_scheduler_type="cosine", optim="paged_adamw_8bit",
        per_device_train_batch_size=1, gradient_accumulation_steps=1,
        num_generations=6, max_prompt_length=256,
        max_completion_length=MAX_SEQ-256,
        max_steps=MAX_STEPS, save_steps=MAX_STEPS,
        report_to="wandb", output_dir="outputs",
    )

    trainer = GRPOTrainer(
        model=model, processing_class=tok,
        reward_funcs=[hard_linecount_reward, hard_meter_reward, hard_rhyme_reward],
        args=args, train_dataset=train_ds,
    )
    trainer.train()

    # ── SAVE & PUSH MERGED ──
    model.save_lora("grpo_saved_lora")
    model.save_pretrained_merged("model", tok, save_method="merged_16bit")
    model.push_to_hub_merged(
        "CK0607/llama3.1-8b-sonnet-rewards-50",
        tok,
        save_method="merged_16bit",
        token=hf_token,
    )

    # ── GENERATE 5 SONNETS ──
    prompts = [
        "Theme: love's endurance",
        "Theme: fleeting beauty",
        "Theme: passage of time",
        "Theme: mortal glory",
        "Theme: the power of art",
    ]
    gen=[]
    model.eval()
    with torch.inference_mode():
        for p in prompts:
            batch = tok(SYS_PROMPT+p, return_tensors="pt").to(model.device)
            out   = model.generate(
                **batch, max_new_tokens=220,
                temperature=0.8, top_p=0.95, do_sample=True
            )
            gen.append(tok.decode(out[0], skip_special_tokens=True))

    # ── WRITE CSV & TXT IN VOLUME ──
    csv_path = f"/root/artifacts/{CSV_NAME}"
    txt_path = f"/root/artifacts/{TXT_NAME}"
    pd.DataFrame({"prompt":prompts, "generated_sonnet":gen}).to_csv(csv_path, index=False)
    with open(txt_path,"w",encoding="utf-8") as f:
        for p,s in zip(prompts, gen):
            f.write(f"{p}\n{s}\n\n")

    # ── PUSH ARTIFACTS TO HF ──
    # api = HfApi(token=hf_token)
    # api.upload_file(csv_path, path_in_repo=CSV_NAME, repo_id=HF_REPO_ID, token=hf_token)
    # api.upload_file(txt_path, path_in_repo=TXT_NAME, repo_id=HF_REPO_ID, token=hf_token)

    # ── COMMIT VOLUME ──
    VOLUME.commit()

    # ── RETURN BYTES FOR LOCAL SAVE ──
    with open(csv_path, "rb") as f: csv_bytes = f.read()
    with open(txt_path, "rb") as f: txt_bytes = f.read()
    return {"csv": csv_bytes, "txt": txt_bytes}

# ─────────────── 4.  LOCAL ENTRYPOINT ───────────────
@stub.local_entrypoint()
def main():
    """Runs the remote training job and saves both artifacts locally."""
    print("🚀 Launching remote GRPO finetune …")
    files = run_train_push.remote()           # returns dict of bytes

    os.makedirs("artifacts_local", exist_ok=True)

    csv_local = os.path.join("artifacts_local", CSV_NAME)
    txt_local = os.path.join("artifacts_local", TXT_NAME)

    with open(csv_local, "wb") as f:
        f.write(files["csv"])
    with open(txt_local, "wb") as f:
        f.write(files["txt"])

    print(f"✅ CSV saved → {csv_local}")
    print(f"✅ TXT saved → {txt_local}")
