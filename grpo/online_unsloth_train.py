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
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from unsloth import FastLanguageModel
import re
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BertTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import logging,datetime
# New imports for on-policy dataset and evaluation
import sys
import time
import random
from typing import List, Dict, Any
from torch.utils.data import Dataset
from vllm import SamplingParams

# Add project root to Python path for internal imports
project_root = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, project_root)

from config import Config, setup_logging
from games import GameByName, Games
from evaluation.evaluator import Evaluator
from agents.vllm_agent import VLLMAgent
from agents.universal_minimax_agent import UniversalMinimaxAgent

directory = os.path.dirname(os.path.abspath(__file__))
# 1. Configuration
model_path = "/remote-home1/share/models/Qwen3-8B"
model_path = os.path.normpath(os.path.join(directory, model_path) if not os.path.isabs(model_path) else model_path)

output_dir = "./outputs/DrCoNi_onpolicy_12000_gen6"

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
# Replaced static JSON dataset loading with an on-policy streaming dataset

def filter_rewards(action_rewards: Dict[str, float]) -> bool:
    """
    Keep only board states with meaningful signal.
    Same logic as in generate_grpo_dataset.py.
    """
    reward_values = list(action_rewards.values())
    if len(reward_values) <= 1:
        return False
    if all(v == reward_values[0] for v in reward_values):
        return False
    win_in_n_threshold = 990
    if any(v > win_in_n_threshold for v in reward_values) or any(v < -win_in_n_threshold for v in reward_values):
        return True
    return False


def _build_entries_from_consolidated(consolidated_path: str, minimax_depth: int = 6) -> List[Dict[str, Any]]:
    """Convert a consolidated evaluation JSON file into GRPO entries."""
    import json
    entries: List[Dict[str, Any]] = []
    processed_boards = set()
    minimax_agent = UniversalMinimaxAgent(max_depth=minimax_depth)

    with open(consolidated_path, "r") as f:
        consolidated_data = json.load(f)

    detailed_logs = consolidated_data.get("detailed_logs", {})
    for game_name, game_data in detailed_logs.items():
        try:
            game_class = GameByName(game_name)
        except KeyError:
            continue
        games = game_data.get("games", {})
        for _gid, game_record in games.items():
            for move_record in game_record.get("moves", []):
                board_str = move_record.get("board_before", "")
                if not board_str:
                    continue
                board_key = f"{game_name}:{board_str}"
                if board_key in processed_boards:
                    continue
                processed_boards.add(board_key)
                # Load game state
                game = game_class()
                try:
                    game.load_state_from_representation(board_str)
                except Exception:
                    continue
                # Compute rewards via minimax
                try:
                    action_rewards = minimax_agent.get_action_rewards(game)
                except Exception:
                    continue
                if not action_rewards or not filter_rewards(action_rewards):
                    continue
                try:
                    prompts = game.get_chat_history_for_llm(minimax_agent)
                except Exception:
                    continue
                entries.append({
                    "prompt": prompts,
                    "task": game_name,
                    "reward_model": {"ground_truth": action_rewards},
                })
    return entries


class OnPolicyGRPODataset(Dataset):
    """
    Dynamic on-policy dataset. When an index beyond current size is requested,
    it triggers self-play rollouts using the current model (via vLLM) and
    appends freshly generated GRPO entries.
    """
    def __init__(
        self,
        model,
        tokenizer,
        games_to_run: List[str] | None = None,
        num_games_per_type: int = 2,
        minimax_depth: int = 6,
        entries_per_refresh: int = 256,
        virtual_size: int = 10_000,
    ) -> None:
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.games_to_run = games_to_run or list(Games.keys())
        self.num_games_per_type = num_games_per_type
        self.minimax_depth = minimax_depth
        self.entries_per_refresh = entries_per_refresh
        self.virtual_size = virtual_size
        self.data: List[Dict[str, Any]] = []
        self.processed_boards: set[str] = set()

        cfg = Config()
        
        # vLLM sampling for agents during self-play
        self.sampling_params = SamplingParams(
            temperature=cfg.TEMPERATURE, top_p=cfg.TOP_P, max_tokens=cfg.MAX_GENERATION_LENGTH, n=1
        )

        # Setup evaluator and logging directory under debug/
        ts = time.strftime("%Y%m%d-%H%M%S")
        self.debug_base = os.path.normpath(os.path.join(project_root, "debug", f"onpolicy_{ts}"))
        os.makedirs(self.debug_base, exist_ok=True)

        cfg.NUM_EVAL_GAMES = self.num_games_per_type
        cfg.OUTPUT_DIR_BASE = self.debug_base
        self.evaluator = Evaluator(cfg)

        # For reward funcs creation convenience
        self.tasks = list(self.games_to_run)

    def __len__(self) -> int:
        # Provide a large virtual size so the trainer keeps sampling batches
        return max(len(self.data), self.virtual_size)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if idx >= len(self.data):
            self._ensure_length(idx + 1)
        return self.data[idx]

    def _ensure_length(self, target_len: int) -> None:
        while len(self.data) < target_len:
            self._rollout_and_append()

    def _rollout_and_append(self) -> None:
        llm_engine = getattr(self.model, "vllm_engine", None)
        if llm_engine is None:
            raise RuntimeError("Model does not expose vllm_engine. Set fast_inference=True in FastLanguageModel.from_pretrained.")
        agent = VLLMAgent(llm_engine, self.sampling_params, self.tokenizer, name="Policy-vLLM")

        subdir = os.path.join(self.debug_base, f"rollout_{int(time.time())}_{random.randint(0, 9999):04d}")
        os.makedirs(subdir, exist_ok=True)

        # --- Balancing logic: estimate entries per game and adjust num_eval_games ---
        # Track how many entries we have per game so far
        game_entry_counts = {g: 0 for g in self.games_to_run}
        for entry in self.data:
            if entry.get("task") in game_entry_counts:
                game_entry_counts[entry["task"]] += 1
        min_count = min(game_entry_counts.values())
        max_count = max(game_entry_counts.values())
        # Target: bring all games up to min_count + entries_per_refresh//len(games)
        target_per_game = min_count + max(1, self.entries_per_refresh // max(1, len(self.games_to_run)))

        # For each game, estimate how many eval games to run to get close to target
        # Use a moving average of entries per eval game (default to 10 if no data yet)
        if not hasattr(self, 'avg_entries_per_eval'):
            self.avg_entries_per_eval = {g: 4 for g in self.games_to_run}
        new_avg_entries = {}
        for game_name in self.games_to_run:
            needed = max(0, target_per_game - game_entry_counts[game_name])
            avg_per_eval = self.avg_entries_per_eval.get(game_name, 4)
            num_eval_games = max(1, int((needed + avg_per_eval - 1) // avg_per_eval))
            new_avg_entries[game_name] = avg_per_eval
            # Prepare config for this game
            cfg = Config()
            cfg.NUM_EVAL_GAMES = num_eval_games
            cfg.OUTPUT_DIR_BASE = subdir
            evaluator = Evaluator(cfg)
            from evaluation.evaluator import ConsolidatedLogger
            consolidated_logger = ConsolidatedLogger(agent.name, agent.name, cfg)
            try:
                game_class = GameByName(game_name)
            except KeyError:
                continue
            try:
                results, eval_logger = evaluator.evaluate_agent_vs_agent_with_logger(
                    agent, agent, game_class, cfg.NUM_EVAL_GAMES
                )
                consolidated_logger.add_game_evaluation_data(game_name, eval_logger, results)
            except Exception:
                continue
            consolidated_path = consolidated_logger.save_consolidated_logs_to_file()
            new_entries = _build_entries_from_consolidated(consolidated_path, self.minimax_depth)
            new_entries = [e for e in new_entries if e.get("task") == game_name]
            if num_eval_games > 0:
                new_avg_entries[game_name] = 0.8 * avg_per_eval + 0.2 * (len(new_entries) / num_eval_games)
            if needed > 0 and new_entries:
                self.data.extend(new_entries[:needed])
        self.avg_entries_per_eval = new_avg_entries


# Instantiate the on-policy dataset
onpolicy_games = ['Connect4','Nim','Dragstone']#list(Games.keys())
dataset = OnPolicyGRPODataset(
    model=None,  # placeholder, will be set after model is created
    tokenizer=None,  # placeholder, will be set after tokenizer is created
    games_to_run=onpolicy_games,
    num_games_per_type=2,
    minimax_depth=6,
    entries_per_refresh=32,
    virtual_size=5000,
)
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
# After model/tokenizer are created, attach them to the on-policy dataset
if isinstance(dataset, OnPolicyGRPODataset):
    dataset.model = model
    dataset.tokenizer = tokenizer
    # Refresh evaluator to ensure debug dir exists (already set in __init__)

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

# Define tasks from on-policy dataset
tasks = dataset.tasks if isinstance(dataset, OnPolicyGRPODataset) else list(set(dataset['task']))
# 4. Custom Reward Function
invalidReward=-2000
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
            if task[i]!=task_name:
                
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
reward_functions=[get_reward_function(task) for task in tasks]

# 5. GRPO Configuration with vLLM
from trl import GRPOTrainer, GRPOConfig
# Clean GRPO config dict to avoid syntax issues
grpo_config_args = {
    'output_dir': output_dir,
    'num_train_epochs': 5,
    'per_device_train_batch_size': 1,
    'gradient_accumulation_steps': 1,
    'learning_rate': 1e-5,
    'optim': 'adamw_8bit',
    'lr_scheduler_type':'constant',
    'num_generations': 6,
    'logging_steps': 10,
    'save_steps': 100,
    'report_to': 'wandb',
    'run_name': output_dir.split('/')[-1],
    'remove_unused_columns': False,  # Keep 'reward_model' column for the reward function
    'max_prompt_length': max_prompt_length,
    'max_completion_length': max_seq_length - max_prompt_length,
}
if FULL_PARAMETER_TRAINING:
    grpo_config_args['generation_kwargs'] = {'max_length': max_seq_length}

# --- vLLM Integration (optional server mode) ---
if VLLM_SERVER_MODE:
    vllm_server_args = {
        'use_vllm': True,
        'vllm_mode': 'server',
        'vllm_server_base_url': 'http://localhost:8001',
        # 'vllm_gpu_memory_utilization': 0.5,
    }
    grpo_config_args.update(vllm_server_args)

grpo_config = GRPOConfig(**grpo_config_args)

# 6. Initialize GRPOTrainer
trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    args=grpo_config,
    train_dataset=dataset,  # Use on-policy dataset
    reward_funcs=reward_functions,
)

# 7. Start Training
print("Starting GRPO training with vLLM...")

trainer.train(resume_from_checkpoint=False)
print("Training finished.")

# 8. Save the final model
trainer.save_model(output_dir)
print(f"Model saved to {output_dir}")
