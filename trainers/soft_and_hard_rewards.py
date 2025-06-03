# grpo_train_modal_bleu_sim.py  ────────────────────────────────────────────
# GRPO finetuning with BLEU-2 + SBERT-cosine soft rewards (no numeric answer).

import os, re, math, modal, torch
from modal import Secret
from datasets import load_dataset, Dataset
import torch.nn.functional as F

# ─── 1. Modal stub, image, volume ─────────────────────────────────────────
stub = modal.App("unsloth_grpo_train")

image = (
    modal.Image.debian_slim()
    .apt_install("git", "curl", "gcc", "libgomp1", "build-essential", "cmake")
    .pip_install(
        "unsloth", "trl", "datasets>=2.19", "tqdm>=4.66",
        "huggingface_hub", "vllm", "wandb",
        "sentence-transformers", "sacrebleu>=2.4.0"
    )
)

VOLUME = modal.Volume.from_name("unsloth-grpo-vol", create_if_missing=True)

# ─── 2. Remote GPU job ────────────────────────────────────────────────────
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
    import sacrebleu                          # BLEU-2
    from sentence_transformers import SentenceTransformer, util

    hf_token = ""
    wandb.login(key="")

    # ─── Model + LoRA setup ───────────────────────────────────────────────
    max_seq_length = 2048
    lora_rank      = 32

    model, tokenizer = FastLanguageModel.from_pretrained(
        "meta-llama/meta-Llama-3.1-8B-Instruct",
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        fast_inference=True,
        max_lora_rank=lora_rank,
        gpu_memory_utilization=0.6,
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

    # ─── Helper functions (extraction) ────────────────────────────────────
    def extract_hash_answer(text: str) -> str | None:
        """Gets the numeric answer after #### in GSM8K-style answers."""
        return text.split("####")[1].strip() if "####" in text else None

    SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""
    def split_reasoning_answer(full: str):
        """Returns reference reasoning (everything before the <answer> tag)."""
        if "<answer>" in full:
            reasoning = full.split("<answer>")[0]
        else:
            reasoning = full
        return reasoning.strip()

    # ─── Build dataset ────────────────────────────────────────────────────
    def get_dataset(split="train") -> Dataset:
        data = load_dataset("CK0607/gsm8k-r1-llama70b", "default")[split]

        def _map(x):
            ref_reasoning = split_reasoning_answer(x["answer"])
            gold_answer   = extract_hash_answer(x["answer"])
            return {
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": x["question"]},
                ],
                "reference_reasoning": ref_reasoning,
                "answer": gold_answer,                # <- for correctness_reward_func
            }

        return data.map(_map, remove_columns=data.column_names)

    dataset = get_dataset()

    # ─── Formatting-related helpers ───────────────────────────────────────
    def extract_xml_answer(text: str) -> str:
        return text.split("<answer>")[-1].split("</answer>")[0].strip()

    def count_xml(text: str) -> float:
        cnt = 0.0
        if text.count("<reasoning>\n")  == 1: cnt += 0.125
        if text.count("\n</reasoning>\n") == 1: cnt += 0.125
        if text.count("\n<answer>\n")   == 1:
            cnt += 0.125
            cnt -= len(text.split("\n</answer>\n")[-1]) * 0.001
        if text.count("\n</answer>")    == 1:
            cnt += 0.125
            cnt -= (len(text.split("\n</answer>")[-1]) - 1) * 0.001
        return cnt

    xml_soft   = re.compile(r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>", re.S)
    xml_strict = re.compile(r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$", re.S)

    # ─── Reward functions ────────────────────────────────────────────────
    def xmlcount_reward_func(completions, **_):
        return [count_xml(c[0]["content"]) for c in completions]

    def soft_format_reward_func(completions, **_):
        return [0.5 if xml_soft.match(c[0]["content"]) else 0.0 for c in completions]

    def strict_format_reward_func(completions, **_):
        return [0.5 if xml_strict.match(c[0]["content"]) else 0.0 for c in completions]

    def correctness_reward_func(completions, answer, **_):
        responses = [c[0]["content"] for c in completions]
        extracted = [extract_xml_answer(r) for r in responses]
        return [2.0 if r == a else 0.0 for r, a in zip(extracted, answer)]

    def int_reward_func(completions, **_):
        extracted = [extract_xml_answer(c[0]["content"]) for c in completions]
        return [0.5 if r.isdigit() else 0.0 for r in extracted]

    # ─── BLEU-2 + SBERT-cosine combo reward ──────────────────────────────
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embedder.eval(); embedder.requires_grad_(False)
    bleu_metric = sacrebleu.metrics.BLEU(smooth_method="exp", effective_order=False)

    def _grab_reasoning(text: str) -> str:
        m = re.search(r"<reasoning>(.*?)</reasoning>", text, re.S)
        return m.group(1).strip() if m else text.strip()

    def _norm(x, μ=0.5, σ=0.25):
        return math.tanh((x - μ) / σ)  # maps to (-1, 1)

    def combo_reward_func(reference_reasoning, completions, **_):
        refs = reference_reasoning
        gens = [_grab_reasoning(c[0]["content"]) for c in completions]

        # SBERT cosine similarity
        with torch.no_grad():
            ref_emb = embedder.encode(refs, convert_to_tensor=True, normalize_embeddings=True)
            gen_emb = embedder.encode(gens, convert_to_tensor=True, normalize_embeddings=True)
            cos = util.cos_sim(gen_emb, ref_emb).diagonal().cpu().tolist()

        rewards = []
        for r, g, c in zip(refs, gens, cos):
            bleu = bleu_metric.corpus_score([g], [[r]]).score / 100.0  # 0-1
            rewards.append(0.7 * c + 0.3 * _norm(bleu))
        return rewards

    # ─── GRPO config & trainer ───────────────────────────────────────────
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
        max_grad_norm               = 0.1,
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
            combo_reward_func,
            correctness_reward_func,
            int_reward_func,               # ← we keep these hard rewards
        ],
        args             = training_args,
        train_dataset    = dataset,
    )

    trainer.train()

    # ─── Save & push ─────────────────────────────────────────────────────
    model.save_lora("grpo_saved_lora")
    model.save_pretrained_merged("model", tokenizer, save_method="merged_16bit")
    model.push_to_hub_merged(
        "CK0607/unsloth-trained-qwen-bleu-sim-250-softandhard",
        tokenizer,
        save_method="merged_16bit",
        token=hf_token,
    )
    VOLUME.commit()
