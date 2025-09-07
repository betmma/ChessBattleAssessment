# ==== PPO + vLLM SERVER (OpenAI API) one-step pipeline for your A/B/C flow ====
# Start vLLM server first (example):
#   export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
#   CUDA_VISIBLE_DEVICES=0,1 vllm serve /remote-home1/share/models/Qwen3-8B --host 0.0.0.0 --port 8000 --dtype auto --api-key token-abc123 --enable-lora --max-loras 8 --max-lora-rank 32 --max_model_len 24000 --data-parallel-size 2
# curl -X POST http://localhost:8000/v1/load_lora_adapter \
#   -H "Content-Type: application/json" \
#   -H "Authorization: Bearer token-abc123" \
#   -d '{
#     "lora_name": "ppo_adapter",
#     "lora_path": "ChessBattleAssessment/asymmetric/ppo_lora_adapter"
#   }'

#
# Python deps:
# conda create -n assym python=3.10 -y
#   pip install "trl==0.9.6" "transformers>=4.43" "accelerate" "peft" "torch" "openai>=1.35" "requests" "vllm" "bitsandbytes" "aenum"
#   "vllm<0.10" "transformers<4.54.0"
#   "deepspeed"
# CUDA_VISIBLE_DEVICES=1,3 accelerate launch --config_file zero1.yaml test.py
#  A) Generate a Game class (Prompt A)
#  B) Build N_goals goal states via multi-turn proposing (Prompt B)
#  C) For each goal, run N_tries solver attempts (Prompt C)
# Rewards:
#  - Phase C: each round in a try gets 1 if goal reached, else 0
#  - Phase B: each round in building that goal gets (1 - p) if p>0 else 0, where p is success rate from C
#  - Phase A: single reward = variance of success rates over the N_goals
#

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import re
import copy
import json
import time,datetime
import torch
import random
import tempfile
import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import List, Any, Optional, Tuple, Dict, Literal
import heapq, openai



from accelerate import Accelerator
acc = Accelerator()  # makes AcceleratorState() initialized on every rank

# If running with DeepSpeed, make sure micro-batch is set even if you won't pass a dataloader.
if acc.state.deepspeed_plugin is not None:
    ds_cfg = acc.state.deepspeed_plugin.deepspeed_config
    # Provide sane defaults if missing
    ds_cfg.setdefault("train_micro_batch_size_per_gpu", 1)
    ds_cfg.setdefault("gradient_accumulation_steps", 1)

import requests
from openai import OpenAI
from transformers import AutoTokenizer
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead
import bitsandbytes as bnb
from peft import LoraConfig

from gameFilter import filterGame, FilterGameResult
import enum

START_TIME_STR=datetime.datetime.now().strftime('_%Y%m%d-%H%M%S')
# ----------------------------
# Config: server + model
# ----------------------------
BASE_MODEL = "/remote-home1/share/models/Qwen3-8B"  # your training policy base
VLLM_URL = os.getenv("VLLM_URL", "http://localhost:8000")
VLLM_KEY = os.getenv("VLLM_KEY", "token-abc123")
ADAPTER_NAME = "ppo_adapter"

client = OpenAI(base_url=f"{VLLM_URL}/v1", api_key=VLLM_KEY)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token

# vLLM sampling params equivalent
SAMPLING_EXTRA = {"top_p": 0.95, "top_k": 20}
TEMPERATURE = 0.6
MAX_TOKENS = 16384  # you can raise this, keep server limits in mind
PPO_MAX_TOKENS = 5500
BATCH_GEN_LIMIT = 64  # max contexts per generation batch

@dataclass
class RunConfig:
    K_moves: int = 5
    N_goals: int = 5
    N_tries: int = 5
    thinking: bool = True

class Phase(enum.Enum):
    A = 'A'
    B = 'B'
    C = 'C'
# --------------------------------------
# PPO policy/value model with PEFT LoRA
# --------------------------------------
ppo_cfg = PPOConfig(
    model_name=BASE_MODEL,
    learning_rate=1e-5,
    mini_batch_size=1,
    batch_size=1,
)
peft_cfg = LoraConfig(
    r=32, lora_alpha=64, 
    target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",]  # adjust to your model
)

policy = AutoModelForCausalLMWithValueHead.from_pretrained(
    BASE_MODEL,
    peft_config=peft_cfg,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None
)
# reduce activation memory
policy.pretrained_model.config.use_cache = False  # must be off for checkpointing
policy.pretrained_model.gradient_checkpointing_enable()

# (Optional, if your stack supports it) use FlashAttention 2
try:
    policy.pretrained_model.config.attn_implementation = "flash_attention_2"
except Exception:
    raise
optimizer = bnb.optim.Adam8bit(policy.parameters(), lr=ppo_cfg.learning_rate)
ppo = PPOTrainer(config=ppo_cfg, model=policy, tokenizer=tokenizer, optimizer=optimizer)

# Where we write the up-to-date LoRA adapter so vLLM can load it
ADAPTER_DIR = tempfile.mkdtemp(prefix="ppo_lora_")
# Permanent adapter directory (saved less frequently)
PERM_ADAPTER_DIR = os.path.join(os.path.dirname(__file__), "ppo_lora_adapter")
os.makedirs(PERM_ADAPTER_DIR, exist_ok=True)
PERM_SAVE_INTERVAL = int(os.getenv("PERM_SAVE_INTERVAL", "50"))  # every N PPO updates
ppo_update_count = 0  # counts successful PPO updates

LOG_FILE = "logs/ppo_rollout_log.jsonl"

def log_event(event: dict):
    """Append a JSON event with timestamp to log file."""
    event_with_time = {"ts": time.time(), "time":datetime.datetime.now().strftime('%Y%m%d-%H%M%S'),**event}
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(event_with_time, ensure_ascii=False) + "\n")
# ========== vLLM server LoRA management ==========
def _post(path: str, payload: dict):
    r = requests.post(
        f"{VLLM_URL}{path}",
        json=payload,
        headers={"Authorization": f"Bearer {VLLM_KEY}", "Content-Type": "application/json"},
        timeout=6000,
    )
    if not r.ok:
        raise RuntimeError(f"POST {path} failed {r.status_code}: {r.text}")
    return r.json() if r.headers.get("content-type","").startswith("application/json") else r.text

def unload_lora_adapter(lora_name: str):
    try:
        return _post("/v1/unload_lora_adapter", {"lora_name": lora_name})
    except Exception as e:
        # ignore if not loaded
        return {"status": "unloaded_or_missing"}

def load_lora_adapter(lora_name: str, lora_path: str):
    return _post("/v1/load_lora_adapter", {"lora_name": lora_name, "lora_path": lora_path})

def hot_reload_lora(lora_name: str, lora_path: str):
    unload_lora_adapter(lora_name)
    return load_lora_adapter(lora_name, lora_path)

def sync_lora_to_disk_for_vllm():
    # Save PEFT adapter weights so vLLM can load them
    ppo.model.pretrained_model.save_pretrained(ADAPTER_DIR)
    hot_reload_lora(ADAPTER_NAME, ADAPTER_DIR)

# Ensure adapter is present at startup so first generation does not 404
sync_lora_to_disk_for_vllm()

# ========== Generation via server ==========
import asyncio, json, httpx
from openai import AsyncOpenAI

# Make sure the HTTP pool can actually hold your concurrency
http_client = httpx.AsyncClient(
    limits=httpx.Limits(
        max_connections=BATCH_GEN_LIMIT * 4,
        max_keepalive_connections=BATCH_GEN_LIMIT * 2
    ),
    timeout=httpx.Timeout(6000)
)

aclient = AsyncOpenAI(
    base_url=f"{VLLM_URL}/v1",
    api_key=VLLM_KEY,
    http_client=http_client
)
async def _gen_one(ctx, thinking=True):
    prompt = tokenizer.apply_chat_template(
        ctx.messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=thinking,
    )
    ctx.last_prompt_text = prompt

    # quick length guard
    if len(tokenizer(prompt, add_special_tokens=False).input_ids) > 7500:
        ctx.last_response_text = "[Error in generation]"
        return ctx

    try:
        resp = await aclient.completions.create(
            model=ADAPTER_NAME,
            prompt=prompt,          # <<< single prompt per request
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            extra_body=SAMPLING_EXTRA,
        )
        ctx.last_response_text = resp.choices[0].text
    except Exception as e:
        ctx.last_response_text = f"[Error in generation: {type(e).__name__}]"
    return ctx



def to_ids(s: str):
    return tokenizer(s, return_tensors="pt", add_special_tokens=False).input_ids[0].to(ppo.accelerator.device)

# ----------------------------
# Prompt Templates (A/B/C)
# ----------------------------
BOARD_GAME_CODE = '''from abc import abstractmethod
from typing import List, Any, Optional, Tuple

class AbstractSystem():
    __slots__ = ['board'] # Can't add other attributes
    def __init__(self):
        super().__init__()
        self.board = self.create_initial_board()

    @abstractmethod
    def create_initial_board(self) -> list:
        """
        Creates and returns the initial board configuration.
        """
        pass
    
    @abstractmethod
    def get_legal_moves(self) -> List[Any]:
        """Return all currently legal moves."""
        pass

    @abstractmethod
    def execute_move(self, move: Any) -> None:
        """Executes a move. self.board will be changed."""
        pass
'''

PROMPT_A_TMPL = """Write an abstract system that inherits AbstractSystem. The system has a board to store some values and some moves to change values. The board should contain all information of current state, and you cannot set new attributes. You cannot use random module. A "proposer" AI will start from the initial board and make a series of valid moves. The final board state becomes the "goal state." A "solver" AI is then given the initial board, the goal state, and your game code. The primary goal is to make the game difficult for "solver" to find a path (sequential moves) from initial state to goal state. The AbstractSystem class code is as below:\n```\n{BOARD_GAME_CODE}\n```\n Your output must ONLY include the class of your system, beginning with:\n```\nfrom typing import List, Any, Optional, Tuple\nclass System(AbstractSystem):```\nDo NOT repeat the AbstractSystem class. Do NOT write game you knew before. Focus on: when can a value be changed, what operation can be used to change a value. Add new operation instead of adding constraints. System does not need to have any practical meaning, instead chaos is good. THINK QUICKLY"""

PROMPT_B_FIRST_TMPL = """Game code:\n```\n{GAME_CODE}\n```\nYou are generating a challenging but solvable goal state of this game by sequentially playing moves. After which the board will be the goal state for "solver". "Solver" will have the game code and the final state you created, and their task is to make moves to reach the goal state from initial state. You should keep their success rate as low as possible but greater than 0.\nRemaining moves you may still play: {remaining_moves}\nNote that in each reply you can only choose one move, but you can continue in further replies if remaining moves is larger than 1.\nLegal moves: {legal}\nCurrent board:\n{board}\nYou can add some explanation or plan in your response, but must end your response with "#### Move chosen\n X" (without quotes), where X is one of the legal moves exactly as an element appearing in the list above, or DONE to finish early."""

PROMPT_B_CONT_TMPL = """Remaining moves you may still play: {remaining_moves}\nLegal moves: {legal}\nCurrent board:\n{board}\nEnd your response with "#### Move chosen\n X" (without quotes), where X is one of the legal moves exactly as an element appearing in the list above, or DONE to finish early."""

PROMPT_C_FIRST_TMPL = """Game code:\n```\n{GAME_CODE}\n```\nYou task is to sequentially play moves to reach goal state.\nRemaining moves you may still play: {remaining_moves}\nNote that in each reply you can only choose one move, but you can continue in further replies if remaining moves is larger than 1.\nLegal moves: {legal}\nCurrent board:\n{board}\nGoal board:\n{goal_board}\nYou can add some explanation or plan in your response, but must end your response with "#### Move chosen\n X" (without quotes), where X is one of the legal moves exactly as an element appearing in the list above."""

PROMPT_C_CONT_TMPL = """Remaining moves you may still play: {remaining_moves}\nLegal moves: {legal}\nCurrent board:\n{board}\nGoal board:\n{goal_board}\nEnd your response with "#### Move chosen\n X" (without quotes), where X is one of the legal moves exactly as an element appearing in the list above."""
# ----------------------------
# Helpers for multi-turn games
# ----------------------------
def extract_game(text: str) -> Optional[str]:
    code_blocks= re.findall(r"```(?:python)?\n(.*?)\n```", text, re.DOTALL | re.IGNORECASE)
    if code_blocks:
        return code_blocks[-1].strip()
    return text.split('</think>')[-1].strip()

def extract_move(text: str) -> Optional[str]:
    if '</think>' not in text: # trimmed
        return None
    move = text.split('#### Move chosen')[-1].strip()
    try:
        move=eval(move)
        return move
    except Exception:
        return move
    
def remove_think(text: str) -> str:
    '''remove thinking part for constructing multi turn chat'''
    return text.split('</think>')[-1].strip()

def run_game_code_and_get_class(game_src: str):
    """
    Exec the generated Game class in a minimal namespace that already defines AbstractSystem.
    Returns Game class object. If not found or doesn't pass check, return None
    """
    namespace = {}
    exec(BOARD_GAME_CODE, namespace, namespace)
    try:
        exec(game_src, namespace, namespace)
    except Exception as e:
        print(f'Error executing game code: {e}')
        return None
    Game = namespace.get("System", None)
    if Game is None:
        print("Generated code did not define class System.")
        return None
    return Game

def messages_for_prompt_A(MOVES: int):
    sys = {"role": "system", "content": "You are a helpful assistant."}
    user = {"role": "user", "content": PROMPT_A_TMPL.format(BOARD_GAME_CODE=BOARD_GAME_CODE)}
    return [sys, user]

def msg_B(game_code: str, game_obj, remaining_moves: int, first: bool):
    legal = game_obj.get_legal_moves()
    board_str = repr(game_obj.board)
    if first:
        txt = PROMPT_B_FIRST_TMPL.format(GAME_CODE=game_code, remaining_moves=remaining_moves, legal=legal, board=board_str)
    else:
        txt = PROMPT_B_CONT_TMPL.format(remaining_moves=remaining_moves, legal=legal, board=board_str)
    return {"role": "user", "content": txt}

def msg_C(game_code: str, game_obj, goal_board, remaining_moves: int, first: bool):
    legal = game_obj.get_legal_moves()
    board_str = repr(game_obj.board)
    goal_str = repr(goal_board)
    if first:
        txt = PROMPT_C_FIRST_TMPL.format(GAME_CODE=game_code, remaining_moves=remaining_moves, legal=legal, board=board_str, goal_board=goal_str)
    else:
        txt = PROMPT_C_CONT_TMPL.format(remaining_moves=remaining_moves, legal=legal, board=board_str, goal_board=goal_str)
    return {"role": "user", "content": txt}

# ----------------------------
# The ONE full PPO step
# ----------------------------

@dataclass
class PhaseAContext:
    phase: str  # 'A'
    messages: List[Dict]
    K_moves: int
    N_goals: int
    N_tries: int
    thinking: bool
    game_code: Optional[str] = None
    GameClass: Optional[Any] = None
    reward_idx: Optional[int] = None
    goal_success_rates: List[Optional[float]] = None
    b_sample_indices_per_goal: List[List[int]] = None
    goals_completed: int = 0
    last_prompt_text: str = ''
    last_response_text: str = ''

    def handle(self, enqueue, add_sample, finalize_reward):
        self.reward_idx = add_sample(self.last_prompt_text, self.last_response_text, None)
        self.game_code = extract_game(self.last_response_text)
        self.GameClass = run_game_code_and_get_class(self.game_code) if self.game_code else None
        if self.GameClass is None:
            finalize_reward([self.reward_idx], -1.5,Phase.A)
            return
        filterResult=filterGame(self.GameClass,self.game_code)
        if filterResult!=FilterGameResult.PASS:
            finalize_reward([self.reward_idx],filterResult.value,Phase.A)
            return
        for gi in range(self.N_goals):
            proposer_game = self.GameClass()
            first_msg = msg_B(self.game_code, proposer_game, remaining_moves=self.K_moves, first=True)
            messages = [{"role": "system", "content": "You are a helpful assistant."}, first_msg]
            bctx = PhaseBContext(
                phase='B', messages=messages, thinking=self.thinking, GameClass=self.GameClass,
                proposer_game=proposer_game, moves_left=self.K_moves, goal_index=gi, parent_A=self,
                b_sample_indices=[], first_turn=True, N_tries=self.N_tries
            )
            enqueue(bctx)

@dataclass
class PhaseBContext:
    phase: str  # 'B'
    messages: List[Dict]
    thinking: bool
    GameClass: Any
    proposer_game: Any
    moves_left: int
    goal_index: int
    parent_A: PhaseAContext
    b_sample_indices: List[int]
    first_turn: bool
    N_tries: int
    last_prompt_text: str = ''
    last_response_text: str = ''

    def handle(self, enqueue, add_sample, finalize_reward):
        idx = add_sample(self.last_prompt_text, self.last_response_text, None)
        self.b_sample_indices.append(idx)
        self.parent_A.b_sample_indices_per_goal[self.goal_index].append(idx)
        chosen = extract_move(self.last_response_text)
        # print(f'[Goal {self.goal_index+1}/{self.parent_A.N_goals} | Moves left {self.moves_left}] Chosen move: {chosen}')
        if chosen is None:
            return
        legal = self.proposer_game.get_legal_moves()
        # print(f'legal moves: {legal}')
        if chosen != 'DONE':
            if str(chosen) not in legal and chosen not in legal:
                chosen = 'DONE'
        if chosen != 'DONE':
            self.proposer_game.execute_move(chosen)
            self.moves_left -= 1
        if self.moves_left > 0 and chosen != 'DONE':
            new_messages = self.messages + [
                {"role": "assistant", "content": remove_think(self.last_response_text)},
                msg_B(self.parent_A.game_code, self.proposer_game, remaining_moves=self.moves_left, first=False)
            ]
            enqueue(PhaseBContext(
                phase='B', messages=new_messages, thinking=self.thinking, GameClass=self.GameClass,
                proposer_game=self.proposer_game, moves_left=self.moves_left, goal_index=self.goal_index,
                parent_A=self.parent_A, b_sample_indices=self.b_sample_indices, first_turn=False, N_tries=self.N_tries
            ))
        elif self.first_turn: # if first_turn and DONE, the state is initial state
            pass
            # print(f'Warning: Goal {self.goal_index+1} proposer finished immediately with DONE on first turn. Raw text:\n{self.last_response_text[-200:]}')
        else:
            goal_board = copy.deepcopy(self.proposer_game.board)
            successes_ref = {"count": 0, "tries": 0}
            for _ in range(self.N_tries):
                solver_game = self.GameClass()
                c_first_msg = msg_C(self.parent_A.game_code, solver_game, goal_board, remaining_moves=self.parent_A.K_moves, first=True)
                messages = [{"role": "system", "content": "You are a helpful assistant."}, c_first_msg]
                enqueue(PhaseCContext(
                    phase='C', messages=messages, thinking=self.thinking, GameClass=self.GameClass,
                    solver_game=solver_game, goal_board=goal_board, moves_left=self.parent_A.K_moves,
                    goal_index=self.goal_index, parent_A=self.parent_A, try_sample_indices=[], successes_ref=successes_ref
                ))

@dataclass
class PhaseCContext:
    phase: str  # 'C'
    messages: List[Dict]
    thinking: bool
    GameClass: Any
    solver_game: Any
    goal_board: Any
    moves_left: int
    goal_index: int
    parent_A: PhaseAContext
    try_sample_indices: List[int]
    successes_ref: Dict[str, int]
    last_prompt_text: str = ''
    last_response_text: str = ''

    def handle(self, enqueue, add_sample, finalize_reward):
        sample_idx = add_sample(self.last_prompt_text, self.last_response_text, None)
        self.try_sample_indices.append(sample_idx)
        chosen = extract_move(self.last_response_text)
        if chosen is None:
            return
        legal = self.solver_game.get_legal_moves()
        if (chosen is None) or (chosen not in legal and str(chosen) not in legal):
            chosen = None
        if chosen is not None:
            self.solver_game.execute_move(chosen)
        self.moves_left -= 1
        if self.moves_left > 0 and self.solver_game.board != self.goal_board:
            new_messages = self.messages + [
                {"role": "assistant", "content": remove_think(self.last_response_text)},
                msg_C(self.parent_A.game_code, self.solver_game, self.goal_board, remaining_moves=self.moves_left, first=False)
            ]
            enqueue(PhaseCContext(
                phase='C', messages=new_messages, thinking=self.thinking, GameClass=self.GameClass,
                solver_game=self.solver_game, goal_board=self.goal_board, moves_left=self.moves_left,
                goal_index=self.goal_index, parent_A=self.parent_A, try_sample_indices=self.try_sample_indices,
                successes_ref=self.successes_ref
            ))
        else:
            success = (self.solver_game.board == self.goal_board)
            if success:
                self.successes_ref["count"] += 1
            self.successes_ref["tries"] += 1
            r_try = 1.0 if success else 0.0
            finalize_reward(self.try_sample_indices, r_try, Phase.C)
            if self.successes_ref["tries"] == self.parent_A.N_tries:
                p = self.successes_ref["count"] / float(self.parent_A.N_tries)
                self.parent_A.goal_success_rates[self.goal_index] = p
                r_b = (1.0 - p) if p > 0.0 else 0.0
                finalize_reward(self.parent_A.b_sample_indices_per_goal[self.goal_index], r_b, Phase.B)
                if all(g is not None for g in self.parent_A.goal_success_rates):
                    var_p = float(np.var(np.array(self.parent_A.goal_success_rates), ddof=0))
                    finalize_reward([self.parent_A.reward_idx], var_p*10, Phase.A)

# ----------------------------
# Queue-based infinite PPO loop (removed one_ppo_step)
# ----------------------------

cfg = RunConfig()

# Containers for samples (text + rewards) kept around; could be truncated periodically
query_texts: List[str] = []
response_texts: List[str] = []
rewards: List[Optional[torch.Tensor]] = []  # None until finalized
unsent_indices: List[int] = []  # indices whose rewards known but not yet sent to PPO

def add_sample(prompt_text: str, response_text: str, reward_value: Optional[float]):
    idx = len(rewards)
    query_texts.append(prompt_text)
    response_texts.append(response_text)
    if reward_value is None:
        rewards.append(None)
    else:
        t = torch.tensor(reward_value, device=ppo.accelerator.device)
        rewards.append(t)
        unsent_indices.append(idx)
        log_event({"prompt_text": prompt_text, "response_text": response_text, "reward": float(t.item())})
    return idx

def finalize_reward(idx_list: List[int], value: float, phase: Phase):
    t = torch.tensor(float(value), device=ppo.accelerator.device)
    for idx in idx_list:
        if rewards[idx] is None:
            rewards[idx] = t
            unsent_indices.append(idx)
            log_event({"prompt_text": query_texts[idx], "response_text": response_texts[idx], "phase": phase.value, "reward": float(value)})

def maybe_run_ppo():
    while len(unsent_indices) >= ppo.config.batch_size:
        batch_indices = unsent_indices[:ppo.config.batch_size]
        if not batch_indices:
            return
        # Remove selected indices from unsent list
        remaining = unsent_indices[len(batch_indices):]
        unsent_indices.clear()
        unsent_indices.extend(remaining)
        query_ids = [to_ids(query_texts[i]) for i in batch_indices]
        resp_ids = [to_ids(response_texts[i]) for i in batch_indices]
        batch_rewards = [rewards[i] for i in batch_indices]
        # Filter out overly long sequences (> PPO_MAX_TOKENS tokens)
        filtered = []
        removed = []
        # Only iterate over the selected batch items, not all items
        batch_query_texts = [query_texts[i] for i in batch_indices]
        batch_response_texts = [response_texts[i] for i in batch_indices]
        for qt, rt, qi, ri, rw, idx in zip(batch_query_texts, batch_response_texts, query_ids, resp_ids, batch_rewards, batch_indices):
            query_token_count = qi.numel()
            resp_token_count = ri.numel()
            isRemoved=False
            if query_token_count + resp_token_count <= PPO_MAX_TOKENS:
                filtered.append((qi, ri, rw, idx))
            else:
                isRemoved=True
                removed.append({"idx": idx, "query_tokens": int(query_token_count), "resp_tokens": int(resp_token_count)})
            log_event({
                "phase": "FILTER", 
                "query_text": qt,  
                "resp_text": rt,    
                "actual_query_tokens": int(query_token_count),
                "actual_resp_tokens": int(resp_token_count),
                "isRemoved": isRemoved
            })
        if not filtered:
            return
        query_ids, resp_ids, batch_rewards, batch_indices = zip(*filtered)
        stats = ppo.step(list(query_ids), list(resp_ids), list(batch_rewards))
        log_event({"phase": "PPO", "stats": {k: (float(v) if isinstance(v, (int, float)) else str(v)) for k, v in stats.items()}, "num_samples": len(filtered)})
        # Permanent save every PERM_SAVE_INTERVAL updates
        global ppo_update_count
        ppo_update_count += 1
        if ppo_update_count%32==0: # load on every query will exceed max-loras and error
            ppo.model.pretrained_model.save_pretrained(ADAPTER_DIR)
            hot_reload_lora(ADAPTER_NAME, ADAPTER_DIR)
        if ppo_update_count % PERM_SAVE_INTERVAL == 0:
            path=PERM_ADAPTER_DIR+'/'+START_TIME_STR+f'/{ppo_update_count}'
            ppo.model.pretrained_model.save_pretrained(path)
            log_event({"phase": "PERM_SAVE", "update": ppo_update_count, "path": path})

# Priorities: C first, then B, then A (like your heap)
PHASE_PRIO = {'C': 0, 'B': 1, 'A': 2}

async def run_workers_and_feed(cfg):
    pq = asyncio.PriorityQueue()   # (prio, seq, ctx)
    seq = 0
    pq_lock = asyncio.Lock()       # protects seq

    # enqueue() used by contexts (A/B/C) to schedule followups
    async def enqueue(ctx):
        nonlocal seq
        pr = PHASE_PRIO.get(getattr(ctx, 'phase', 'A'), 3)
        async with pq_lock:
            await pq.put((pr, seq, ctx))
            seq += 1

    # seed at start
    active_As = []
    async def new_phase_A():
        a = PhaseAContext(
            phase='A',
            messages=messages_for_prompt_A(MOVES=cfg.K_moves),
            K_moves=cfg.K_moves,
            N_goals=cfg.N_goals,
            N_tries=cfg.N_tries,
            thinking=cfg.thinking,
            goal_success_rates=[None]*cfg.N_goals,
            b_sample_indices_per_goal=[[] for _ in range(cfg.N_goals)],
        )
        active_As.append(a)
        await enqueue(a)

    # keep the queue topped up so workers never idle
    async def feeder():
        while True:
            # If low backlog, add more A to spawn B/C work later
            if pq.qsize() < max(BATCH_GEN_LIMIT * 2, BATCH_GEN_LIMIT // 2):
                await new_phase_A()
            await asyncio.sleep(0.005)

    # Worker: always one prompt per HTTP request
    ppo_lock = asyncio.Lock()  # serialize PPO updates
    async def worker(wid: int):
        while True:
            _, _, ctx = await pq.get()
            try:
                await _gen_one(ctx, thinking=cfg.thinking)
                # After generation, handle -> enqueue followups (B/C) and rewards
                ctx.handle(
                    enqueue=lambda c: asyncio.create_task(enqueue(c)),  # schedule enqueue without blocking
                    add_sample=add_sample,
                    finalize_reward=finalize_reward
                )
                # Run PPO periodically (serialized)
                async with ppo_lock:
                    maybe_run_ppo()
                # optional: logging
                with open('logs/generations.log','a',encoding='utf-8') as f:
                    f.write(json.dumps(
                        {"time":datetime.datetime.now().strftime('%Y%m%d-%H%M%S'), "prompt": ctx.last_prompt_text, "response": ctx.last_response_text},
                        ensure_ascii=False
                    ) + "\n")
            finally:
                pq.task_done()

    # spin up workers + feeder
    feeders = [asyncio.create_task(feeder())]
    workers = [asyncio.create_task(worker(i)) for i in range(BATCH_GEN_LIMIT)]

    # Optionally, wait forever (Ctrl+C to stop), or run for N steps:
    await asyncio.gather(*workers, *feeders)
    
async def main():
    await run_workers_and_feed(cfg)

if __name__ == "__main__":
    asyncio.run(main())