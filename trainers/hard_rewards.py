# grpo_train_modal_full.py
# Fixes the Modal volume constructor error and keeps everything else intact.

import os, re, modal, torch
from modal import Secret
from datasets import load_dataset, Dataset
# from unsloth import FastLanguageModel
# from trl import GRPOConfig, GRPOTrainer

# ───────────────────────────────────────────────────────────
# 1.  Modal stub, image, and volume
# ───────────────────────────────────────────────────────────
stub = modal.App("unsloth_grpo_train")

image = (
    modal.Image.debian_slim()
    .apt_install("git", "curl", "gcc", "libgomp1", "build-essential", "cmake")
    .pip_install(
        "unsloth", "trl", "datasets>=2.19", "tqdm>=4.66", "huggingface_hub","vllm","wandb"
    )
)

# Persisted volume (constructor fixed)
VOLUME = modal.Volume.from_name("unsloth-grpo-vol", create_if_missing=True)

# ───────────────────────────────────────────────────────────
# 2.  Remote GPU job – finetune then push to HF
# ───────────────────────────────────────────────────────────
@stub.function(
    image=image,
    gpu="A100",
    timeout=72000,
    volumes={"/root/artifacts": VOLUME},  # keyword updated from shared_volumes
)
def run_train_push():
    """Finetune Unsloth Llama-3-8B with GRPO and push weights to Hugging Face."""
    from unsloth import FastLanguageModel
    from trl import GRPOConfig, GRPOTrainer
    import wandb
    # Retrieve HF token (hard-coded or via Secret)
    hf_token = "hf_token_here"
    wandb.login(key = "wandb_api_key_here")
    # ─── Model + LoRA setup ───────────────────────────────────────────────
    max_seq_length = 512
    lora_rank = 32

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="meta-llama/meta-Llama-3.1-8B-Instruct",
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
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=lora_rank,
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    # ─── Dataset helpers ─────────────────────────────────────────────────
    SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

    XML_COT_FORMAT = """\
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>
"""

    def extract_xml_answer(text: str) -> str:
        answer = text.split("<answer>")[-1].split("</answer>")[0]
        return answer.strip()

    def extract_hash_answer(text: str) -> str | None:
        return text.split("####")[1].strip() if "####" in text else None

    def get_gsm8k_questions(split="train") -> Dataset:
        data = load_dataset("CK0607/gsm8k-r1-llama70b", "default")[split]
        return data.map(
            lambda x: {
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": x["question"]},
                ],
                "answer": extract_hash_answer(x["answer"]),
            }
        )

    dataset = get_gsm8k_questions()

    # ─── Reward functions (unchanged) ────────────────────────────────────
    def correctness_reward_func(prompts, completions, answer, **kwargs):
        responses = [c[0]["content"] for c in completions]
        extracted_responses = [extract_xml_answer(r) for r in responses]
        return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]

    def int_reward_func(completions, **kwargs):
        responses = [c[0]["content"] for c in completions]
        extracted_responses = [extract_xml_answer(r) for r in responses]
        return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]

    def strict_format_reward_func(completions, **kwargs):
        pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
        responses = [c[0]["content"] for c in completions]
        return [0.5 if re.match(pattern, r) else 0.0 for r in responses]

    def soft_format_reward_func(completions, **kwargs):
        pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
        responses = [c[0]["content"] for c in completions]
        return [0.5 if re.match(pattern, r) else 0.0 for r in responses]

    def count_xml(text) -> float:
        cnt = 0.0
        if text.count("<reasoning>\n") == 1:
            cnt += 0.125
        if text.count("\n</reasoning>\n") == 1:
            cnt += 0.125
        if text.count("\n<answer>\n") == 1:
            cnt += 0.125
            cnt -= len(text.split("\n</answer>\n")[-1]) * 0.001
        if text.count("\n</answer>") == 1:
            cnt += 0.125
            cnt -= (len(text.split("\n</answer>")[-1]) - 1) * 0.001
        return cnt

    def xmlcount_reward_func(completions, **kwargs):
        contents = [c[0]["content"] for c in completions]
        return [count_xml(c) for c in contents]

    max_prompt_length = 256

    # ─── GRPO config & trainer ───────────────────────────────────────────
    training_args = GRPOConfig(
        learning_rate=5e-6,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",
        logging_steps=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        num_generations=6,
        max_prompt_length=max_prompt_length,
        max_completion_length=max_seq_length - max_prompt_length,
        max_steps=250,
        save_steps=10,
        max_grad_norm=0.1,
        report_to="wandb",
        output_dir="outputs",
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            xmlcount_reward_func,
            soft_format_reward_func,
            strict_format_reward_func,
            int_reward_func,
            correctness_reward_func,
        ],
        args=training_args,
        train_dataset=dataset,
    )

    trainer.train()

    # ─── Save & push to Hub ──────────────────────────────────────────────
    model.save_lora("grpo_saved_lora")

    # Merge to 16-bit
    model.save_pretrained_merged(
        "model", tokenizer, save_method="merged_16bit"
    )
    model.push_to_hub_merged(
        "model_name_here",
        tokenizer,
        save_method="merged_16bit",
        token=hf_token,
    )

    # Persist artefacts
    VOLUME.commit()
