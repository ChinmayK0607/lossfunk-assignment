from datasets import load_dataset, Dataset
import random
from huggingface_hub import login

# 1. Login to Hugging Face
# <insert hftoken>
from datasets import load_dataset, Dataset
import random
from huggingface_hub import login


login(token=hf_token)  # Replace with your actual token


dataset = load_dataset("openai/gsm8k", "main", split="train")


total_indices = list(range(len(dataset)))

=
random.seed(42)
first_1000_indices = random.sample(total_indices, 1000)


random.seed(1337)
remaining_indices = list(set(total_indices) - set(first_1000_indices))
new_1000_indices = random.sample(remaining_indices, 1000)

t
new_1000_dataset = dataset.select(new_1000_indices)


new_1000_dataset.push_to_hub("CK0607/gsm8k-1000-extra", private=True,token = hf_token)
