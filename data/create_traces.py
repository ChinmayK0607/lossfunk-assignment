#

import re
import time
import pandas as pd
from groq import Groq
from datasets import load_dataset

# ───────────────────────────────────────────────────────────────────────────────
# 1. Helper functions
# ───────────────────────────────────────────────────────────────────────────────

def extract_hash_answer(text: str) -> str | None:
    """
    Given a string like "... #### 1280", return "1280".
    If no "####" marker is found, return None.
    """
    if "####" in text:
        # split on "####" and take everything after, strip whitespace
        return text.split("####", maxsplit=1)[1].strip()
    return None

import re

def extract_model_answer(full_response: str) -> str | None:
    """
    Try to pull out the model’s final answer in one of two forms:
      1) Anything inside <answer>...</answer>
      2) Anything inside a \\boxed{...} (e.g. **Final Answer:** \\[ \\boxed{8800} \\])
    Returns the inner text (e.g. "8800") if found, or None otherwise.
    """
    # 1) Try <answer>...</answer>
    match = re.search(r"<answer>(.*?)</answer>", full_response, re.DOTALL)
    if match:
        return match.group(1).strip()

    # 2) Fallback: look for \boxed{...}
    #    We allow optional whitespace and optional surrounding LaTeX delimiters (\[ \], $$, etc.),
    #    but the core is \boxed{...}.
    match_boxed = re.search(r"\\boxed\{\s*([^}]+?)\s*\}", full_response)
    if match_boxed:
        return match_boxed.group(1).strip()

    return None


def extract_model_reasoning(full_response: str) -> str | None:
    """
    From the model’s complete response (which should contain <think>...</think>),
    return the text inside the <think> tags. If no tags found, return None.
    """
    match = re.search(r"<think>(.*?)</think>", full_response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

def answers_equal(model_ans: str, ground_ans: str) -> bool:
    """
    Normalize both strings by removing commas and converting to float/int when possible.
    If neither conversion works, fall back to a case‐insensitive string compare.
    """
    def normalize(s: str):
        s = s.replace(",", "").strip()
        # Try int first
        try:
            return int(s)
        except ValueError:
            pass
        # Try float next
        try:
            return float(s)
        except ValueError:
            pass
        # Fallback to stripped lowercase string
        return s.lower()

    return normalize(model_ans) == normalize(ground_ans)


dataset_dict = load_dataset("openai/gsm8k", "main")
dataset = dataset_dict["train"].shuffle(seed=42).select(range(1000))  # e.g. if your split is called "train"


resume_choice = input("Resume from last checkpoint? (y/n): ").strip().lower()
if resume_choice == "y":
    try:
        # Attempt to read last checkpoint index
        with open("checkpoint.txt", "r") as f:
            last_idx = int(f.read().strip())
        # Load existing correct‐reasoning CSV into df_correct
        df_correct = pd.read_csv("reasoning_dataset.csv", dtype=str)
        start_idx = last_idx + 1
        print(f"Resuming from index {start_idx} (last checkpoint at {last_idx}).")
    except FileNotFoundError:
        print("No checkpoint found. Starting from scratch.")
        df_correct = pd.DataFrame(columns=["question", "reasoning", "answer"])
        start_idx = 0
else:
    # Start fresh, ignore any existing checkpoint
    print("Starting from scratch.")
    df_correct = pd.DataFrame(columns=["question", "reasoning", "answer"])
    start_idx = 0



client = Groq(api_key="<groq api token>")



for idx, row in enumerate(dataset):
    # Skip rows until we reach start_idx (if resuming)
    if idx < start_idx:
        continue

    # Each `row` is a dict-like object with keys "question" and "answer"
    question_text = row["question"].strip()
    ground_truth_raw = row["answer"]
    ground_truth = extract_hash_answer(ground_truth_raw)  # e.g. "1280", or None if malformed

    if ground_truth is None:
        # If the ground-truth answer column doesn’t contain "####", skip this row
        print(f"[{idx}] Skipping because no '####' found in answer: {ground_truth_raw!r}")
        # Even though we skip, we should still checkpoint counts every 5 processed indices
    else:
        # 4.a. Send the prompt to Groq with reasoning_format="raw" and no streaming (parsed)
        print(f"[{idx}] Sending to model: {question_text!r}")
        prompt = (
            "Enclose the final answer in strictly <answer> </answer> tags inside your reasoning, "
            "in plaintext and not latex, do not add any slashes or anything else before the number—just the numbers. "
            "Always enclose the answer in <answer></answer> tags. "
        )
        completion = client.chat.completions.create(
            model="deepseek-r1-distill-llama-70b",
            messages=[{"role": "user", "content": prompt + question_text}],
            temperature=0.6,
            max_completion_tokens=1024,
            top_p=0.95,
            reasoning_format="parsed"
        )

        # Accumulate the single (parsed) response into full_response
        full_response = ""
        try:
            # parsed format returns the entire answer + reasoning in one object
            delta_content = completion.choices[0].message.content
            print(delta_content)
            if delta_content:
                full_response = delta_content
        except Exception as e:
            print(f"    ➔ Error fetching response for row {idx}: {e}")
            # Skip adding to df_correct, but continue to next index
            full_response = ""

        # 4.b. Extract the model's final answer and its reasoning trace
        model_answer = extract_model_answer(full_response)
        model_reasoning = extract_model_reasoning(full_response)

        if model_answer is None:
            print(f"    ➔ Row {idx}: Model did not return <answer> tags. Skipping.")
        else:
            # 4.c. Compare the model's answer to the ground truth
            if answers_equal(model_answer, ground_truth):
                # They match after normalizing commas and ".0"
                print(f"    ✅ Row {idx}: {model_answer!r} matches ground truth {ground_truth!r}.")
                final_answer_with_hash = f"#### {model_answer}"
                combined_answer = full_response + "\n" + final_answer_with_hash

                # 4.d. If correct, add to df_correct
                new_row = {
                    "question": question_text,
                    "reasoning": model_reasoning or "",
                    "answer": combined_answer
                }
                df_correct = pd.concat([df_correct, pd.DataFrame([new_row])], ignore_index=True)
            else:
                print(f"    ❌ Row {idx}: Model answer {model_answer!r} ≠ ground truth {ground_truth!r}. Skipping.")

    
    if (idx - start_idx + 1) % 5 == 0:
        # Write the current df_correct to CSV
        df_correct.to_csv("reasoning_dataset.csv", index=False)
        # Write out the last processed index to checkpoint.txt
        with open("checkpoint.txt", "w") as cf:
            cf.write(str(idx))
        print(f"    — Checkpoint saved at row {idx}. Wrote reasoning_dataset.csv and checkpoint.txt.")

  ]
    time.sleep(0.2)

# ───────────────────────────────────────────────────────────────────────────────
# 5. Final save of the “correct reasoning” dataset to CSV (if not already done)
# ───────────────────────────────────────────────────────────────────────────────

# One last write in case the last loop did not hit a 5-step boundary
df_correct.to_csv("reasoning_dataset.csv", index=False)
from datasets import Dataset
import pandas as pd

# # Load your CSV
df = pd.read_csv("reasoning_dataset.csv")

# # Convert to HF dataset
dataset = Dataset.from_pandas(df)

# # Push to the Hub
dataset.push_to_hub("CK0607/gsm8k-r1-llama70b")
with open("checkpoint.txt", "w") as cf:
    cf.write(str(idx))
print(f"\nDone. Final CSV written with {len(df_correct)} rows. Last processed index = {idx}.")
