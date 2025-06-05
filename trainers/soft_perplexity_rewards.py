# grpo_train_modal_full.py  ──────────────────────────────────────────────
# GRPO with *soft* perplexity reward (no numeric-answer checks).

import os, re, modal, torch
from modal import Secret
from datasets import load_dataset, Dataset

# ───────────────────────────────────────────────────────────
# 1.  Modal stub, image, volume               (↑ add dependency)
# ───────────────────────────────────────────────────────────
stub = modal.App("unsloth_grpo_train")

image = (
    modal.Image.debian_slim()
    .apt_install("git", "curl", "gcc", "libgomp1", "build-essential", "cmake")
    .pip_install(
        "unsloth", "trl", "datasets>=2.19", "tqdm>=4.66",
        "huggingface_hub", "vllm", "wandb",
        "sentence-transformers"               # ← NEW
    )
)

VOLUME = modal.Volume.from_name("unsloth-grpo-vol", create_if_missing=True)

# ───────────────────────────────────────────────────────────
# 2.  Remote GPU job – finetune & push (unchanged except rewards)
# ───────────────────────────────────────────────────────────
@stub.function(
    image=image,
    gpu="A100",
    timeout=72000,
    volumes={"/root/artifacts": VOLUME},
)
def run_train_push():
    from unsloth import FastLanguageModel
    from trl import GRPOConfig, GRPOTrainer
    import wandb
    from sentence_transformers import SentenceTransformer, util
    import torch.nn.functional as F
    
    hf_token = "hf_token_here"
    wandb.login(key="wandb_api_key_here")

    # — Model + LoRA set-up (identical) —
    max_seq_length = 2048
    lora_rank      = 32

    model, tokenizer = FastLanguageModel.from_pretrained(
        "meta-llama/meta-Llama-3.1-8B-Instruct",
        max_seq_length=max_seq_length,
        load_in_4bit=True, fast_inference=True,
        max_lora_rank=lora_rank, gpu_memory_utilization=0.6,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,
        target_modules=[
            "q_proj","k_proj","v_proj","o_proj",
            "gate_proj","up_proj","down_proj",
        ],
        lora_alpha=lora_rank,
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    # —————  Dataset helpers —————
    SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

    def extract_hash_answer(text: str) -> str | None:
        return text.split("####")[1].strip() if "####" in text else None

    def split_reasoning_answer(full: str) -> tuple[str, str]:
        """Returns (reasoning, answer_text)."""
        if "<answer>" in full:
            reasoning = full.split("<answer>")[0]
            answer    = full.split("<answer>")[-1]
        else:                    # fall-back
            reasoning, answer = "", full
        return reasoning.strip(), answer.strip()

    def get_gsm8k_questions(split="train") -> Dataset:
        data = load_dataset("CK0607/gsm8k-r1-llama70b", "default")[split]

        def _map(x):
            reasoning, _ = split_reasoning_answer(x["answer"])
            return {
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": x["question"]},
                ],
                "reference_reasoning": reasoning,
            }
        return data.map(_map, remove_columns=data.column_names)

    dataset = get_gsm8k_questions()

    # —————  Reward functions —————
    # (A) -- formatting / XML rewards (unchanged) -------------
    def count_xml(text: str) -> float:
        cnt = 0.0
        if text.count("<reasoning>\n") == 1:          cnt += 0.125
        if text.count("\n</reasoning>\n") == 1:       cnt += 0.125
        if text.count("\n<answer>\n") == 1:
            cnt += 0.125
            cnt -= len(text.split("\n</answer>\n")[-1]) * 0.001
        if text.count("\n</answer>") == 1:
            cnt += 0.125
            cnt -= (len(text.split("\n</answer>")[-1]) - 1) * 0.001
        return cnt

    xml_pattern_soft   = re.compile(r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>", re.S)
    xml_pattern_strict = re.compile(r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$", re.S)

    def xmlcount_reward_func(completions, **_):
        return [count_xml(c[0]["content"]) for c in completions]

    def soft_format_reward_func(completions, **_):
        return [0.5 if xml_pattern_soft.match(c[0]["content"]) else 0.0
                for c in completions]

    def strict_format_reward_func(completions, **_):
        return [0.5 if xml_pattern_strict.match(c[0]["content"]) else 0.0
                for c in completions]

    # (B) -- NEW similarity reward (sentence-embedding cosine) -----
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embedder.eval() 
    embedder.requires_grad_(False)

    def _grab_reasoning(text: str) -> str:
        m = re.search(r"<reasoning>(.*?)</reasoning>", text, re.S)
        return m.group(1).strip() if m else text.strip()

    def similarity_reward_func(reference_reasoning, completions, **_):
        refs = reference_reasoning                       # list[str] from dataset
        gens = [_grab_reasoning(c[0]["content"]) for c in completions]

        with torch.no_grad():
            ref_emb = embedder.encode(refs, convert_to_tensor=True, normalize_embeddings=True)
            gen_emb = embedder.encode(gens, convert_to_tensor=True, normalize_embeddings=True)
            sims    = util.cos_sim(gen_emb, ref_emb).diagonal()  # cosine for each pair
        return sims.cpu().tolist()                       # higher = better

    # ––– FIXED perplexity reward function ––––––––––––––––––
    def _avg_nll(model, tokenizer, prompt_txt: str, reasoning_txt: str,
                 max_len: int = 2048):
        """
        Returns average NLL over (prompt + <reasoning> + reasoning_txt),
        truncating so total length ≤ max_len.
        """
        seq = prompt_txt + "<reasoning>" + reasoning_txt

        # 1) Tokenize + truncate - FIXED: proper handling of special tokens
        toks = tokenizer(
            seq,
            add_special_tokens=True,  # Let tokenizer handle special tokens properly
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
            padding=False
        )
        
        input_ids = toks["input_ids"].to(model.device)
        
        # Need at least 2 tokens for input/target pairs
        if input_ids.size(1) < 2:
            return torch.tensor(0.0, device=model.device)

        # 2) Prepare inputs and targets
        inputs = input_ids[:, :-1]
        targets = input_ids[:, 1:]

        with torch.no_grad():
            outputs = model(inputs)
            logits = outputs.logits  # shape: (batch, seq_len, vocab_size)

        # 3) FIXED: Ensure targets are within valid range
        vocab_size = logits.size(-1)
        # Clamp targets to valid range (shouldn't be necessary with proper tokenization, but safety check)
        targets = torch.clamp(targets, 0, vocab_size - 1)
        
        # 4) Compute average NLL with proper ignore_index
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else -100,
            reduction="mean",
        )
        return loss

    def perplexity_reward_func(prompts, completions,
                               reference_reasoning,  # list[str]
                               model=model, tokenizer=tokenizer, **_):
        rewards = []
        
        for p, comp, ref_r in zip(prompts, completions, reference_reasoning):
            try:
                # Convert prompt messages to text
                prompt_txt = ""
                for msg in p:
                    if msg.get("role") == "system":
                        prompt_txt += f"<|start_header_id|>system<|end_header_id|>\n{msg['content']}<|eot_id|>"
                    elif msg.get("role") == "user":
                        prompt_txt += f"<|start_header_id|>user<|end_header_id|>\n{msg['content']}<|eot_id|>"
                    elif msg.get("role") == "assistant":
                        prompt_txt += f"<|start_header_id|>assistant<|end_header_id|>\n{msg['content']}<|eot_id|>"
                
                # Add assistant start for completion
                prompt_txt += "<|start_header_id|>assistant<|end_header_id|>\n"
                
                gen_r = _grab_reasoning(comp[0]["content"])
                
                # Skip if reasoning is empty
                if not gen_r.strip() or not ref_r.strip():
                    rewards.append(0.0)
                    continue

                nll_gen = _avg_nll(model, tokenizer, prompt_txt, gen_r)
                nll_ref = _avg_nll(model, tokenizer, prompt_txt, ref_r)

                # Handle potential NaN or inf values
                if torch.isnan(nll_gen) or torch.isnan(nll_ref) or torch.isinf(nll_gen) or torch.isinf(nll_ref):
                    rewards.append(0.0)
                    continue

                delta = nll_ref - nll_gen
                reward = torch.tanh(delta).item()
                rewards.append(reward)
                
            except Exception as e:
                print(f"Error in perplexity_reward_func: {e}")
                rewards.append(0.0)  # Fallback reward

        return rewards

    # —————  GRPO config & trainer (unchanged) —————
    training_args = GRPOConfig(
        learning_rate               = 5e-6,
        adam_beta1                  = 0.9,
        adam_beta2                  = 0.99,
        weight_decay                = 0.1,
        warmup_ratio                = 0.1,
        lr_scheduler_type           = "cosine",
        optim                       = "paged_adamw_8bit",
        logging_steps               = 1,
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 1,
        num_generations             = 6,
        max_prompt_length           = 256,
        max_completion_length       = max_seq_length - 256,
        max_steps                   = 250,
        save_steps                  = 10,
        max_grad_norm              = 0.1,
        report_to                   = "wandb",
        output_dir                  = "outputs",
    )

    trainer = GRPOTrainer(
        model            = model,
        processing_class = tokenizer,
        reward_funcs     = [
            xmlcount_reward_func,
            soft_format_reward_func,
            strict_format_reward_func,
            # similarity_reward_func,  # Can re-enable this if needed
            perplexity_reward_func,      # ← FIXED soft reward
        ],
        args             = training_args,
        train_dataset    = dataset,
    )

    trainer.train()

    # —————  Save & push (unchanged) —————
    model.save_lora("grpo_saved_lora")
    model.save_pretrained_merged("model", tokenizer, save_method="merged_16bit")
    model.push_to_hub_merged(
        "model_name_here",
        tokenizer,
        save_method="merged_16bit",
        token=hf_token,
    )
    VOLUME.commit()
