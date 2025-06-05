# Lossfunk Assignment


Following is the submission and brief description of how to use the above scripts. 

Disclaimer: Since I was constantly iterating and updating the experiments, there may be issues with running the scripts due to missed dependencies or something. Please excuse that. Also I have taken help of AI with respect to iterating fast and writing code, so please ignore comments and quality of the code.  

Environment: Recommended to use a linux environment. I have used lightning.ai for renting VMs for both ease of use and smooth operation. 

## Setup
1. Clone the repo 
2. `pip install uv`
3. `uv pip install -r requirements.txt`
4. Run `modal setup`. reference -> [docs](https://modal.com/docs/guide)

This will create a modal.toml that will contain your modal secrets.

## Use of scripts


### For Data preparation: 
That should be fine
1. `cd data`
2. run the scripts with the command `python script_name.py`
   

### For training: 


Before running these scripts, please check for hf_token and wandb api keys to be added to the script so as to run it properly on serverless environments and save and push your models. 
Please beware of model names and other variable names that you have to set before running the scripts. 
1. `cd trainers`
2. modal run script_name.py


### For benchmarking

Similar to earlier, check for local variables like Groq Api keys that need to be set so as to have a seamless experience. 
1. `cd benchmarks`
2. For normal benchmarking use `modal run script_name.py` (Beware of model names that you set to benchmark the scripts as well as then names of the final csvs)
3. For llm as a judge benchmarks, use `python script_name.py <file_path_here>`
4. For accuracy, if testing normal accuracy using `accuracy.py`, then use `python accuracy.py <file_path_here>`
5. Else please use `python passat8accuracy.py --base_dir <directory_path_here>` 


