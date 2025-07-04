# MASTER_PORT=29501 CUDA_VISIBLE_DEVICES=5 trl vllm-serve --model "Qwen3-8B" --port 8001 --max_model_len 2048
# torchrun --nproc_per_node=1 train.py
# accelerate launch --config_file deepspeed_zero3.yaml train.py

# export MASTER_PORT=51216 
# accelerate launch --num_processes=2 train.py
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
os.environ['MASTER_PORT']='51218'
import re
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BertTokenizer, BitsAndBytesConfig
from trl import GRPOTrainer, GRPOConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import logging,datetime

directory = os.path.dirname(os.path.abspath(__file__))
# 1. Configuration
model_path = "../../Qwen3-8B"
model_path = os.path.normpath(os.path.join(directory, model_path) if not os.path.isabs(model_path) else model_path)
dataset_path = "../evaluation_results_vllm/grpo/grpo8_reformatted.jsonl"
dataset_path = os.path.normpath(os.path.join(directory, dataset_path) if not os.path.isabs(dataset_path) else dataset_path)
output_dir = "./outputs/grpo8_connect4_qwen8b_strict_reformat"

os.makedirs(output_dir, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename=os.path.join(output_dir, f"evaluation_log_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.log"),
    filemode='w'
)
# 2. Load Dataset
try:
    dataset = load_dataset("json", data_files=dataset_path, split="train")
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit()

# 3. Load Tokenizer and Model
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
)

tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=True,)
model = AutoModelForCausalLM.from_pretrained(
    model_path, local_files_only=True, trust_remote_code=True,
    # device_map="auto", # auto 100% crashes, aten.cat.default: got mixed torch.Tensor and DTensor. cuda will cause uneven memory usage
    torch_dtype=torch.bfloat16,
    # quantization_config=quantization_config,
)

# Set pad token if it's not already set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# model = prepare_model_for_kbit_training(model)

# This is the configuration for the LoRA adapters
peft_config = LoraConfig(
    r=16, # Rank of the update matrices. Lower ranks save more memory.
    lora_alpha=32, # Alpha parameter for scaling.
    target_modules=[ # Specify which modules to apply LoRA to.
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# --- 3. Wrap the base model with the PEFT model ---
model = get_peft_model(model, peft_config)

# 4. Custom Reward Function
invalidReward=-2000
def connect4_reward_function(prompts, completions, completion_ids, **reward_kwargs):
    """
    Custom reward function for the Connect 4 environment.

    Args:
        completions (list of str): The list of generated responses from the model.
        ground_truth (list of list of int): The ground truth rewards for each column.
                                            This comes directly from the dataset.

    Returns:
        list of float: A list of rewards for each completion.
    """
    # print(reward_kwargs)
    rewards = []
    for i, completion in enumerate(completions):
        completion=completion[0]['content']
        # The prompt that generated this completion
        current_prompt = prompts[i]

        # --- Logging ---
        logging.info(f"--- Sample {i} in Batch ---")
        logging.info(f"Prompt: {current_prompt}")
        logging.info(f"Completion: {completion}")
        # ---------------
        try:
            reward = invalidReward
            thinkBegin=completion.count('<think>')
            thinkEnd=completion.count('</think>')
            afterThink=completion.split('</think>')[-1]
            # Extract the column number from the model's output (e.g., "[3]")
            match = re.findall(r'\[(\d)\]', afterThink)
            if thinkBegin==1 and thinkEnd==1 and match:
                chosen_column = int(match[-1])
                # The ground_truth for the i-th prompt is at index i
                current_ground_truth = reward_kwargs['reward_model'][i]['ground_truth']
                if 0 <= chosen_column < len(current_ground_truth):
                    reward = float(current_ground_truth[chosen_column])
        except Exception as e:
            print(f"Error processing completion: '{completion}'. Error: {e}")
            reward = invalidReward  # Assign a low reward for any processing error
        logging.info(f"Assigned Reward: {reward}\n")
        rewards.append(reward)
    return rewards

# 5. GRPO Configuration with vLLM
grpo_config = GRPOConfig(
    output_dir=output_dir,
    num_train_epochs=5,
    per_device_train_batch_size=3, # 
    gradient_accumulation_steps=8,
    learning_rate=2e-6,
    num_generations=8,  # batch 4, num 8, completion 1024, 79gb; 2-8-1024-
    logging_steps=10,
    save_steps=100,
    report_to="wandb",
    run_name=output_dir.split('/')[-1],
    remove_unused_columns=False, # Keep 'reward_model' column for the reward function

    # Key change: max_length is removed. Use max_prompt_length and max_completion_length
    max_prompt_length=512,
    max_completion_length=1024,

    generation_kwargs={'max_tokens': 1024},
    
    # --- vLLM Integration ---
    use_vllm=True,
    vllm_mode="server",
    vllm_server_base_url="http://localhost:8003",
    # Adjust based on your GPU memory. 0.3 means 30% of GPU memory is allocated to vLLM.
    # The rest is used for training. You may need to tune this value.
    # vllm_gpu_memory_utilization=0.5,
)

# 6. Initialize GRPOTrainer
trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    args=grpo_config,
    train_dataset=dataset,
    reward_funcs=connect4_reward_function,
)

# 7. Start Training
print("Starting GRPO training with vLLM...")
trainer.train()
print("Training finished.")

# 8. Save the final model
trainer.save_model(output_dir)
print(f"Model saved to {output_dir}")
