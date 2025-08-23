# use vllm server mode: (Upon test model parameters wont update, dont use)
'''
set VLLM_SERVER_MODE=True (below)
unset http_proxy
CUDA_VISIBLE_DEVICES=1 trl vllm-serve --model "/remote-home1/share/models/Qwen3-8B" --port 8001 --max_model_len 16384 --data_parallel_size 1 --enable_prefix_caching True
'''
# use multi gpu: (TCP client failed to connect/validate to host 10.176.58.107:53217 - timed out (try=1, timeout=600000ms))
'''
unset http_proxy
set param "ddp_find_unused_parameters" to true
MASTER_ADDR=127.0.0.1 MASTER_PORT=29500 NCCL_SOCKET_IFNAME=lo NCCL_NET=Socket NCCL_IB_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4   --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:29500 -m test_unsloth_train.py --backend gloo

torchrun --nproc_per_node=4 launcher.py test_unsloth_train.py 
'''
# python unsloth_train.py
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

from unsloth import FastLanguageModel
import re
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BertTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import logging,datetime

directory = os.path.dirname(os.path.abspath(__file__))
# 1. Configuration
model_path = "/remote-home1/yrmou/models/DrCoNi_lv2_12000-ckpt-1900"
model_path = os.path.normpath(os.path.join(directory, model_path) if not os.path.isabs(model_path) else model_path)
dataset_path = "/remote-home1/yrmou/ChessBattleAssessment/evaluation_results_vllm/grpo/DrCoNi_lv3_raw_d6_balanced.jsonl"
dataset_path = os.path.normpath(os.path.join(directory, dataset_path) if not os.path.isabs(dataset_path) else dataset_path)
output_dir = "./outputs/DrCoNi_lv3_12000_gen6"

FULL_PARAMETER_TRAINING = False
VLLM_SERVER_MODE = False

if FULL_PARAMETER_TRAINING:
    output_dir += "_full"

os.makedirs(output_dir, exist_ok=True)

# copy this script to output_dir for record
import shutil
shutil.copy(__file__, os.path.join(output_dir, os.path.basename(__file__)+'-'+datetime.datetime.now().strftime('_%Y%m%d-%H%M%S')+'.backup'))

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
def clean_ground_truth(example):
    # Access the messy dictionary
    messy_dict = example['reward_model']['ground_truth']

    # Create a new dictionary containing only the key-value pairs
    # where the value is not None.
    cleaned_dict = {k: v for k, v in messy_dict.items() if v is not None}

    # Update the example
    example['reward_model']['ground_truth'] = cleaned_dict
    return example
dataset=dataset.map(clean_ground_truth)
tasks=list(set(dataset['task']))

# 3. Load Tokenizer and Model
max_seq_length = 12000 # Can increase for longer reasoning traces
max_prompt_length = 800 # Maximum length of the prompt
lora_rank = 32
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_path,
    max_seq_length = max_seq_length,
    load_in_4bit = False, # 'NoneType' object has no attribute 'absmax' if true. unsloth issue #2910
    load_in_8bit= False, # RuntimeError: CUDA driver error: invalid argument
    fast_inference = False if VLLM_SERVER_MODE else True, # Enable vLLM fast inference
    full_finetuning = FULL_PARAMETER_TRAINING, # full parameter 4bit + 2000 max length still oom 
    max_lora_rank = lora_rank,
    gpu_memory_utilization = 0.4, # Reduce if out of memory 0.7->0.6 speed increased by x2?
)

if FULL_PARAMETER_TRAINING:
    class dummyWith:
        def __init__(self):
            pass

        def __enter__(self):
            pass

        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

    model.disable_adapter = dummyWith # For compatibility with peft, which uses model.disable_adapter() to disable adapters during inference
else:
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
MULTIPLE_REWARD_FUNCTIONS=False # i thought if using different functions for each game, trl can log rewards for each game. but it seems that only reward function that returns value for all instances (like format reward that applies to all) can be logged. so this should be False, otherwise these will just be NaN loggings
def get_reward_function(task_name):
    def reward_function(prompts, completions, task, completion_ids, **reward_kwargs):
        """
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
            if task[i]!=task_name and MULTIPLE_REWARD_FUNCTIONS:
                rewards.append(None)
                continue

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
                    if re.match(r'\[\d\]|\(\d\)',afterThink):
                        # This is a valid completion format, e.g., "[0]"
                        afterThink = afterThink[1:-1]
                    current_ground_truth = reward_kwargs['reward_model'][i]['ground_truth']
                    # seems that dataset load forces each ground_truth dict to contain all keys appeared in whole dataset, like this {'ground_truth': {'(0, 0)': None, '(0, 1)': -997.0, '(0, 2)': None, '(1, 0)': None, '(1, 1)': -997.0, '(1, 2)': None, '(2, 0)': None, '(2, 1)': 996.0, '(2, 2)': -997.0, '(1, 3)': None, '(3, 1)': None, '(3, 2)': None, '(3, 3)': None, '(0, 3)': None, '(0, 4)': None, '(0, 5)': None, '(0, 6)': None, '(0, 7)': None ... '0': None, '2': None, '4': None, '6': None, '1': None, '3': None, '5': None}}]} so need to remove None values
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
    reward_function.__name__ = f"{task_name}_reward_function"
    return reward_function
if MULTIPLE_REWARD_FUNCTIONS:
    reward_functions = [get_reward_function(task) for task in tasks]
else:
    reward_functions = [get_reward_function('games')]

# 5. GRPO Configuration with vLLM
from trl import GRPOTrainer, GRPOConfig
grpo_config_args={
    'output_dir':output_dir,
    'num_train_epochs':5,
    'per_device_train_batch_size':1, # 
    'gradient_accumulation_steps':1,
    'learning_rate':1e-5,
    'optim': "adamw_8bit",
    'lr_scheduler_type':'constant',
    'num_generations':6, 
    'logging_steps':10,
    'save_steps':100,
    'report_to':"wandb",
    'run_name':output_dir.split('/')[-1],
    'remove_unused_columns':False, # Keep 'reward_model' column for the reward function

    # Key change: max_length is removed. Use max_prompt_length and max_completion_length
    'max_prompt_length':max_prompt_length,
    'max_completion_length':max_seq_length - max_prompt_length,

    'generation_kwargs':{'max_length': max_seq_length} if FULL_PARAMETER_TRAINING else None, # full parameter uses it i dunno why
    
    # 'ddp_find_unused_parameters':True # multi gpu

    # use_liger_loss=True, # this slightly increases memory usage (^^;
    
    # --- vLLM Integration ---
    # use_vllm=True,
    # vllm_mode="server",
    # vllm_server_base_url="http://localhost:8004",
    # Adjust based on your GPU memory. 0.3 means 30% of GPU memory is allocated to vLLM.
    # The rest is used for training. You may need to tune this value.
    # vllm_gpu_memory_utilization=0.5,
}

if VLLM_SERVER_MODE:
    vllm_server_args={
        'use_vllm':True,
        'vllm_mode':'server',
        'vllm_server_base_url':"http://localhost:8001"
    }
    grpo_config_args=grpo_config_args|vllm_server_args

grpo_config=GRPOConfig(**grpo_config_args)

# 6. Initialize GRPOTrainer
trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    args=grpo_config,
    train_dataset=dataset,
    reward_funcs=reward_functions,
)

# 7. Start Training
print("Starting GRPO training with vLLM...")

trainer.train(resume_from_checkpoint=False)
print("Training finished.")

# 8. Save the final model
trainer.save_model(output_dir)
print(f"Model saved to {output_dir}")
