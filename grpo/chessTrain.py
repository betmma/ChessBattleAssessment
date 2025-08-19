# python unsloth_train.py
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
from unsloth import FastLanguageModel
import re
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BertTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import logging,datetime

directory = os.path.dirname(os.path.abspath(__file__))
# 1. Configuration
model_path = "/remote-home1/yrmou/models/Qwen3-8B-unsloth-bnb-4bit"
model_path = os.path.normpath(os.path.join(directory, model_path) if not os.path.isabs(model_path) else model_path)

output_dir = "./outputs/lichess_puzzle_16000"

FULL_PARAMETER_TRAINING = False

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
from chessPuzzle.chessPuzzleDataset import ChessPuzzleDataset
csv_path = "/remote-home1/yrmou/chessPuzzle/lichess_db_puzzle.csv"
dataset=ChessPuzzleDataset(csv_path, initial_model_elo=400, k_factor=4, elo_range=100)

# 3. Load Tokenizer and Model
max_seq_length = 16000 # Can increase for longer reasoning traces
max_prompt_length = 800 # Maximum length of the prompt
lora_rank = 32
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_path,
    max_seq_length = max_seq_length,
    load_in_4bit = True, # 'NoneType' object has no attribute 'absmax' if true. unsloth issue #2910
    load_in_8bit= False, # RuntimeError: CUDA driver error: invalid argument
    fast_inference = True, # Enable vLLM fast inference
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
import re, chess 
from typing import List, Dict, Any
CORRECT_REWARD = 1.0
INCORRECT_REWARD = -1.0
INVALID_FORMAT_REWARD = -2.0 # Penalize for not even providing a valid move format

def reward_function(
    prompts: List[str],
    completions: List[str],
    completion_ids: List[int],
    **reward_kwargs: Dict[str, Any]
) -> List[float]:
    """
    Calculates rewards for chess puzzle completions and updates the model's Elo.

    Args:
        completions (list of str): The list of generated responses from the model.
        reward_kwargs (dict): Must contain 'reward_model' from the dataset.

    Returns:
        list of float: A list of rewards for each completion.
    """
    rewards = []

    reward_models = reward_kwargs.get('reward_model', [])

    # Regex to find a UCI move (e.g., e2e4, a7a8q)
    uci_move_pattern = re.compile(r'\b([a-h][1-8][a-h][1-8][qrbn]?)\b')
    # Regex to find a SAN move (captures, promotions, checks/mates, castles)
    san_move_pattern = re.compile(r"\b(?:O-O(?:-O)?[+#]?|[PNBRQK]?[a-h]?[1-8]?x?[a-h][1-8](?:=[NBRQ])?[+#]?)\b")

    for i, completion_dict in enumerate(completions):
        completion_text = completion_dict[0]['content']
        thinkBegin = completion_text.count('<think>')
        thinkEnd = completion_text.count('</think>')
        afterThink = ""
        if thinkBegin == 1 and thinkEnd == 1:
            afterThink = completion_text.split('</think>')[-1].strip().replace(' ', '')

        current_ground_truth = reward_models[i]['ground_truth']
        solution_move = current_ground_truth['solution_move']  # ground truth in UCI
        puzzle_elo = current_ground_truth['puzzle_elo']
        player_fen = current_ground_truth['player_fen']

        logging.info(f"--- Sample {i} in Batch ---")
        logging.info(f"Model Completion: '{completion_text.strip()}'")
        logging.info(f"Ground Truth Move (UCI): {solution_move}, Puzzle Elo: {puzzle_elo}")

        reward = INVALID_FORMAT_REWARD
        score = None

        # 1) Try to extract SAN and convert to UCI
        san_matches = san_move_pattern.findall(afterThink)
        model_move_uci = None
        if san_matches:
            board_for_san = chess.Board(player_fen)
            for san in reversed(san_matches):  # Prefer the last SAN-like token
                try:
                    move_obj = board_for_san.parse_san(san)
                    model_move_uci = move_obj.uci()
                    logging.info(f"Extracted SAN: {san} -> UCI: {model_move_uci}")
                    break
                except Exception:
                    continue

        # 2) If SAN failed, fallback to UCI extraction directly
        if model_move_uci is None:
            uci_matches = uci_move_pattern.findall(afterThink)
            if uci_matches:
                model_move_uci = uci_matches[-1]
                logging.info(f"Extracted UCI: {model_move_uci}")

        if model_move_uci is not None:
            if model_move_uci == solution_move:
                reward = CORRECT_REWARD
                score = 1.0
                logging.info(f"Result: CORRECT. Reward: {reward}")
            else:
                try:
                    board = chess.Board(player_fen)
                    move_obj = chess.Move.from_uci(model_move_uci)
                except chess.InvalidMoveError:
                    move_obj = None

                if move_obj in board.legal_moves:
                    board.push(move_obj)
                    if board.is_checkmate():
                        reward = CORRECT_REWARD
                        score = 1.0
                        logging.info(f"Result: CORRECT (alternative checkmate found). Reward: {reward}")
                    else:
                        reward = INCORRECT_REWARD
                        score = 0.0
                        logging.info(f"Result: INCORRECT (legal move but not a winning one). Reward: {reward}")
                else:
                    reward = INCORRECT_REWARD
                    score = 0.0
                    logging.info(f"Result: INCORRECT (illegal move for this position). Reward: {reward}")
        else:
            logging.info(f"Result: INVALID FORMAT. Reward: {reward}")

        # Update the model's Elo rating via the dataset instance. Skip on invalid format
        if score is not None:
            dataset.update_elo(puzzle_elo, score)
        rewards.append(reward)

    return rewards


# 5. GRPO Configuration with vLLM
from trl import GRPOTrainer, GRPOConfig
grpo_config = GRPOConfig(
    output_dir=output_dir,
    num_train_epochs=1,
    per_device_train_batch_size=2, # 
    gradient_accumulation_steps=1, # unsloth bug, must be 1
    learning_rate=1e-5,
    lr_scheduler_type='constant',
    optim = "adamw_8bit",
    num_generations=2,  
    generation_batch_size=8,
    mask_truncated_completions=True,
    logging_steps=5,
    save_steps=100,
    report_to="wandb",
    run_name='chessPuzzle/'+output_dir.split('/')[-1],
    remove_unused_columns=False, # Keep 'reward_model' column for the reward function

    # Key change: max_length is removed. Use max_prompt_length and max_completion_length
    max_prompt_length=max_prompt_length,
    max_completion_length=max_seq_length - max_prompt_length,

    generation_kwargs={'max_length': max_seq_length} if FULL_PARAMETER_TRAINING else None, # full parameter uses it i dunno why
    
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

class EloLoggingGRPOTrainer(GRPOTrainer):
    """
    A custom GRPOTrainer that logs the model's Elo rating to wandb.
    """
    def log(self, logs: Dict[str, float], start_time) -> None:
        """
        Overrides the log method to add the current model Elo to the logs.
        This method is called by the trainer at each logging step.
        """
        # The trainer stores the train_dataset instance as an attribute.
        # We can access it to get the current Elo.
        if self.train_dataset and hasattr(self.train_dataset, "model_elo"):
            # Add the elo to the dictionary of logs.
            # Using a prefix like "elo/" helps group charts in wandb.
            logs["elo/model_elo"] = self.train_dataset.model_elo
        
        # Call the parent's log method to handle the actual logging
        # to the console, wandb, etc.
        super().log(logs, start_time)
        

trainer = EloLoggingGRPOTrainer(
    model=model,
    processing_class=tokenizer,
    args=grpo_config,
    train_dataset=dataset,
    reward_funcs=reward_function,
)

# 7. Start Training
print("Starting GRPO training with vLLM...")

trainer.train(resume_from_checkpoint=False)
print("Training finished.")

# 8. Save the final model
trainer.save_model(output_dir)
print(f"Model saved to {output_dir}")
