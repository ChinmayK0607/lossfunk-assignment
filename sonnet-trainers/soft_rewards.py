#!/usr/bin/env python
"""
grpo_llama3_pplx_sonnets.py
────────────────────────────────────────────────────────────────────────────
GRPO finetune on Shakespeare sonnets with a *perplexity-delta* reward:
  reward = tanh( NLL_ref  −  NLL_gen )
A higher reward is given when the model’s generated *reasoning / poem*
has lower NLL (i.e. higher likelihood) than the reference sonnet.
The script pushes the merged checkpoint and stores 5 sample sonnets
( CSV + TXT ) both in a Modal volume and locally.
"""
import os, re, math, modal, torch, pandas as pd
from datasets import load_dataset
import torch.nn.functional as F
from functools import partial

# ─────────────── 0.  CONSTANTS / TOKENS ───────────────
hf_token   = "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"
HF_REPO_ID = "CK0607/unsloth-trained-qwen-ce-perplexity"   # artefact repo for CSV/TXT
WANDB_KEY  = "702c80eb1559b9d775b36ee98b23300b6863bf09"

GPU_TYPE   = "A100"
MAX_SEQ    = 512
LORA_RANK  = 32
MAX_STEPS  = 50

CSV_NAME   = "pplx_sonnets_samples.csv"
TXT_NAME   = "pplx_sonnets_samples.txt"

# ─────────────── 1.  MODAL APP / IMAGE / VOLUME ───────────────
stub = modal.App("grpo_llama3_pplx_sonnets")

image = (
    modal.Image.debian_slim()
    .apt_install("git", "curl", "gcc", "libgomp1", "build-essential", "cmake")
    .pip_install(
        "unsloth", "trl", "datasets", "tqdm",
        "huggingface_hub", "vllm", "pandas", "wandb"
    )
)

VOLUME = modal.Volume.from_name("unsloth-grpo-sonnet-vol", create_if_missing=True)

# ─────────────── 2.  REMOTE JOB ───────────────
@stub.function(
    image=image,
    gpu=GPU_TYPE,
    timeout=72000,
    volumes={"/root/artifacts": VOLUME},
)
def run_train_push():
    """
    Finetunes, pushes merged model, writes CSV & TXT to the Modal
    volume, **returns both files as bytes** for local saving.
    """
    import torch, math, re, pandas as pd
    from unsloth import FastLanguageModel
    from trl import GRPOConfig, GRPOTrainer
    from huggingface_hub import HfApi
    import wandb

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
        target_modules=[
            "q_proj","k_proj","v_proj","o_proj",
            "gate_proj","up_proj","down_proj"
        ],
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

    # ── HELPER: grab “reasoning” (here: full assistant completion) ──
    def _grab_reasoning(text: str) -> str:
        # If you use "<reasoning>" tags you can slice; here we return whole text.
        return text

    # ── HELPER: average NLL over (prompt + <reasoning> + reasoning_txt) ──
    def _avg_nll(
        model, tokenizer, prompt_txt: str, reasoning_txt: str, max_len: int = 2048
    ) -> torch.Tensor:
        """
        Returns average negative-log-likelihood of the concatenated sequence
        (prompt_txt + '<reasoning>' + reasoning_txt), truncated to ≤ max_len.
        """
        seq = prompt_txt + "<reasoning>" + reasoning_txt
        toks = tokenizer(
            seq,
            add_special_tokens=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
            padding=False,
        ).to(model.device)

        input_ids = toks["input_ids"]
        if input_ids.size(1) < 2:
            return torch.tensor(0.0, device=model.device)

        inputs  = input_ids[:, :-1]
        targets = input_ids[:, 1:]

        with torch.no_grad():
            logits = model(inputs).logits

        # Clamp just in case
        targets = torch.clamp(targets, 0, logits.size(-1) - 1)

        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=tok.pad_token_id if tok.pad_token_id is not None else -100,
            reduction="mean",
        )
        return loss  # already per-token average

    # ── PERPLEXITY-DELTA REWARD ──
    def perplexity_reward_func(
        prompts,                    # List[List[dict]] messages
        completions,                # List[List[dict]] assistant outputs
        reference_reasoning,        # List[str]
        model=model, tokenizer=tok, **_
    ):
        rewards = []

        for p_msgs, comp, ref_r in zip(prompts, completions, reference_reasoning):
            try:
                # Convert prompt messages → flat prompt text
                prompt_txt = ""
                for msg in p_msgs:
                    if msg["role"] == "system":
                        prompt_txt += (
                            "<|start_header_id|>system<|end_header_id|>\n"
                            + msg["content"]
                            + "<|eot_id|>"
                        )
                    elif msg["role"] == "user":
                        prompt_txt += (
                            "<|start_header_id|>user<|end_header_id|>\n"
                            + msg["content"]
                            + "<|eot_id|>"
                        )
                    elif msg["role"] == "assistant":
                        prompt_txt += (
                            "<|start_header_id|>assistant<|end_header_id|>\n"
                            + msg["content"]
                            + "<|eot_id|>"
                        )

                # Add assistant header for **this** completion
                prompt_txt += "<|start_header_id|>assistant<|end_header_id|>\n"

                gen_r = _grab_reasoning(comp[0]["content"])

                # Empty check
                if not gen_r.strip() or not ref_r.strip():
                    rewards.append(0.0)
                    continue

                nll_gen = _avg_nll(model, tokenizer, prompt_txt, gen_r)
                nll_ref = _avg_nll(model, tokenizer, prompt_txt, ref_r)

                if torch.isnan(nll_gen) or torch.isnan(nll_ref):
                    rewards.append(0.0)
                    continue

                delta   = nll_ref - nll_gen
                reward  = torch.tanh(delta).item()
                rewards.append(reward)

            except Exception as e:
                print(f"[perplexity_reward_func] error: {e}")
                rewards.append(0.0)

        return rewards

    # Curry reference list so GRPOTrainer sees a two-arg callable
    
    perp_reward = partial(
    perplexity_reward_func,
    reference_reasoning=train_ds["reference"],
)

# NEW – give the partial a .__name__ so Unsloth is happy
    perp_reward.__name__ = "perplexity_reward"

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
        model=model,
        processing_class=tok,
        reward_funcs=[perp_reward],       # ← ONLY perplexity reward
        args=args,
        train_dataset=train_ds,
    )
    trainer.train()

    # ── SAVE & PUSH MERGED ──
    model.save_lora("grpo_saved_lora")
    model.save_pretrained_merged("model", tok, save_method="merged_16bit")
    model.push_to_hub_merged(
        "CK0607/llama3.1-8b-sonnet-pplx-50",
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

    # ── OPTIONAL: push artefacts to HF ──
    # api = HfApi(token=hf_token)
    # api.upload_file(csv_path, path_in_repo=CSV_NAME, repo_id=HF_REPO_ID, token=hf_token)
    # api.upload_file(txt_path, path_in_repo=TXT_NAME, repo_id=HF_REPO_ID, token=hf_token)

    # ── COMMIT VOLUME ──
    VOLUME.commit()

    # ── RETURN BYTES FOR LOCAL SAVE ──
    with open(csv_path, "rb") as f: csv_bytes = f.read()
    with open(txt_path, "rb") as f: txt_bytes = f.read()
    return {"csv": csv_bytes, "txt": txt_bytes}

# ─────────────── 3.  LOCAL ENTRYPOINT ───────────────
@stub.local_entrypoint()
def main():
    """Runs the remote training job and saves artefacts locally."""
    print("🚀 Launching remote GRPO finetune …")
    files = run_train_push.remote()

    os.makedirs("artifacts_local", exist_ok=True)
    csv_local = os.path.join("artifacts_local", CSV_NAME)
    txt_local = os.path.join("artifacts_local", TXT_NAME)

    with open(csv_local, "wb") as f:
        f.write(files["csv"])
    with open(txt_local, "wb") as f:
        f.write(files["txt"])

    print(f"✅ CSV saved → {csv_local}")
    print(f"✅ TXT saved → {txt_local}")
