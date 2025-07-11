# MASTER_PORT=29501 CUDA_VISIBLE_DEVICES=5 trl vllm-serve --model "Qwen3-8B" --port 8001 --max_model_len 2048
# torchrun --nproc_per_node=1 train.py
# accelerate launch --config_file deepspeed_zero3.yaml train.py

# export MASTER_PORT=51216 
# accelerate launch --num_processes=2 train.py
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
from unsloth import FastLanguageModel
import re
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BertTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import logging,datetime

directory = os.path.dirname(os.path.abspath(__file__))
# 1. Configuration
model_path = "../../Qwen3-8B"
model_path = os.path.normpath(os.path.join(directory, model_path) if not os.path.isabs(model_path) else model_path)
dataset_path = "../evaluation_results_vllm/grpo/5games_4_filtered.jsonl"
dataset_path = os.path.normpath(os.path.join(directory, dataset_path) if not os.path.isabs(dataset_path) else dataset_path)
output_dir = "./outputs/5games_4_qwen8b_strict_8192"

os.makedirs(output_dir, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename=os.path.join(output_dir, f"evaluation_log_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.log"),
    filemode='w'
)

CHECK_AFTER_APPLY_CHAT_TEMPLATE = True
if CHECK_AFTER_APPLY_CHAT_TEMPLATE:
    import trl.data_utils
    # log raw text after applying chat template
    def maybe_apply_chat_template(
        example: dict[str, list[dict[str, str]]],
        tokenizer,
        tools= None,
    ) -> dict[str, str]:
        if trl.data_utils.is_conversational(example):
            ret= trl.data_utils.apply_chat_template(example, tokenizer, tools)
            logging.info(f"After applying chat template:\n----------------------\n{ret}")
            return ret
        else:
            return example
    trl.data_utils.maybe_apply_chat_template=maybe_apply_chat_template
# 2. Load Dataset
try:
    dataset = load_dataset("json", data_files=dataset_path, split="train")
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit()

# 3. Load Tokenizer and Model
max_seq_length = 16384 # Can increase for longer reasoning traces
lora_rank = 32
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_path,
    max_seq_length = max_seq_length,
    load_in_4bit = False, # False for LoRA 16bit
    fast_inference = True, # Enable vLLM fast inference
    max_lora_rank = lora_rank,
    gpu_memory_utilization = 0.7, # Reduce if out of memory
)

model = FastLanguageModel.get_peft_model(
    model,
    r = lora_rank, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha = lora_rank*2, # *2 speeds up training
    use_gradient_checkpointing = "unsloth", # Reduces memory usage
    random_state = 3407,
)

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
            if thinkBegin==1 and thinkEnd==1:
                afterThink=completion.split('</think>')[-1].strip().replace(' ','')
                if re.match(r'\[\d\]',afterThink):
                    # This is a valid completion format, e.g., "[0]"
                    afterThink = afterThink[1:-1]
                current_ground_truth = reward_kwargs['reward_model'][i]['ground_truth']
                # seems that trl forces each ground_truth dict to contain all keys appeared in whole dataset, like this {'ground_truth': {'(0, 0)': None, '(0, 1)': -997.0, '(0, 2)': None, '(1, 0)': None, '(1, 1)': -997.0, '(1, 2)': None, '(2, 0)': None, '(2, 1)': 996.0, '(2, 2)': -997.0, '(1, 3)': None, '(3, 1)': None, '(3, 2)': None, '(3, 3)': None, '(0, 3)': None, '(0, 4)': None, '(0, 5)': None, '(0, 6)': None, '(0, 7)': None ... '0': None, '2': None, '4': None, '6': None, '1': None, '3': None, '5': None}}]} so need to remove None values
                current_ground_truth = {k.replace(' ',''): v for k, v in current_ground_truth.items() if v is not None}
                # print(current_ground_truth)
                reward = current_ground_truth.get(afterThink, invalidReward)
                logging.info(f"After think: {afterThink[:10]}, Reward Dict: {current_ground_truth}, Reward: {reward}")
            else:
                logging.warning(f"Invalid completion format for sample {i}.")
        except Exception as e:
            print(f"Error processing completion: '{completion}'. Error: {e}")
            print(f'Reward kwargs: {reward_kwargs}')
            reward = invalidReward  # Assign a low reward for any processing error
        logging.info(f"Assigned Reward: {reward}\n")
        rewards.append(reward)
    return rewards

# 5. GRPO Configuration with vLLM
from trl import GRPOTrainer, GRPOConfig
grpo_config = GRPOConfig(
    output_dir=output_dir,
    num_train_epochs=5,
    per_device_train_batch_size=1, # 
    gradient_accumulation_steps=8,
    learning_rate=2e-6,
    num_generations=4,  # this doesn't affect memory. batch 4, prompt 512, completion 1024, 79gb; 3-512-1024- 66gb; 1-1024-4096-72gb
    logging_steps=10,
    save_steps=100,
    report_to="wandb",
    run_name=output_dir.split('/')[-1],
    remove_unused_columns=False, # Keep 'reward_model' column for the reward function

    # Key change: max_length is removed. Use max_prompt_length and max_completion_length
    max_prompt_length=1024,
    max_completion_length=16384,

    generation_kwargs={'max_tokens': 16384},
    
    # use_liger_loss=True, # this slightly increases memory usage (^^;
    
    # --- vLLM Integration ---
    # use_vllm=True,
    # vllm_mode="server",
    # vllm_server_base_url="http://localhost:8004",
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
