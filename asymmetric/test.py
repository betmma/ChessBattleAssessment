# ==== PPO + vLLM SERVER (OpenAI API) one-step pipeline for your A/B/C flow ====
# Start vLLM server first (example):
# !!REMEMBER TO SET BELOW ENV VAR
#   export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
#   CUDA_VISIBLE_DEVICES=0,1,2 vllm serve /remote-home1/share/models/Qwen3-8B --host 0.0.0.0 --port 8000 --dtype auto --api-key token-abc123 --enable-lora --max-loras 8 --max-lora-rank 32 --max_model_len 32000 --data-parallel-size 3

#
# Python deps:
# conda create -n assym python=3.10 -y
#   pip install "trl==0.9.6" "transformers>=4.43" "accelerate" "peft" "torch" "openai>=1.35" "requests" "vllm" "bitsandbytes" "aenum"
#   "vllm<0.10" "transformers<4.54.0"
#   "deepspeed"
# export TOKENIZERS_PARALLELISM=true to stop huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks... warning
# CUDA_VISIBLE_DEVICES=3 accelerate launch --config_file zero1.yaml test.py
#  A) Generate a Game class (Prompt A)
#  B) Build N_goals goal states via multi-turn proposing (Prompt B)
#  C) For each goal, run N_tries solver attempts (Prompt C)
# Rewards:
#  - Phase C: each round in a try gets (1 if goal reached else 0) - p, where p is success rate of N_TRIES Cs
#  - Phase B: each round in building that goal gets (1 - p) if p>0 else 0
#  - Phase A: single reward = variance of success rates over the N_goals * 10
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

from gameFilter import filterGame, FilterGameResult, extract_game, run_game_code_and_get_class
import enum
from settings import (BASE_MODEL, VLLM_URL, VLLM_KEY, KEEP_ACTIVE_ADAPTERS, BATCH_GEN_LIMIT, MAX_TOKENS, PROMPT_MAX_TOKENS, PPO_MAX_TOKENS, TEMPERATURE, SAMPLING_EXTRA, PERM_SAVE_INTERVAL, LOG_FILE, GAME_COMPLETE_LOG, GENERATION_LOG, K_MOVES, N_GOALS, N_TRIES, THINKING, EVAL_PERIOD, EVAL_RESULTS_FILE, EVAL_AT_INIT, REINFORCE_STYLE
)

START_TIME_STR=datetime.datetime.now().strftime('_%Y%m%d-%H%M%S')
# ----------------------------
# Config: server + model (use settings directly)
# ----------------------------
CURRENT_ADAPTER_NAME: Optional[str] = None
_loaded_adapter_names = deque()  # track load order

client = OpenAI(base_url=f"{VLLM_URL}/v1", api_key=VLLM_KEY)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token

class Phase(enum.Enum):
    A = 'A'
    B = 'B'
    C = 'C'
    EVAL = 'EVAL'
# --------------------------------------
# PPO policy/value model with PEFT LoRA
# --------------------------------------
ppo_cfg = PPOConfig(
    model_name=BASE_MODEL,
    learning_rate=1e-5,
    mini_batch_size=1,
    batch_size=1,
)
if REINFORCE_STYLE:
    ppo_cfg.vf_coef=0.0
    ppo_cfg.cliprange_value=0.0
    ppo_cfg.gamma=0.0
    ppo_cfg.lam=0.0

peft_cfg = LoraConfig(
    r=16, lora_alpha=32, 
    target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",]  # adjust to your model
)

policy = AutoModelForCausalLMWithValueHead.from_pretrained(
    BASE_MODEL,
    peft_config=peft_cfg,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None
)
if REINFORCE_STYLE:
    for p in policy.v_head.parameters():
        p.requires_grad = False

# reduce activation memory
policy.pretrained_model.config.use_cache = False  # must be off for checkpointing
policy.pretrained_model.gradient_checkpointing_enable(
    gradient_checkpointing_kwargs={"use_reentrant": False})

# (Optional, if your stack supports it) use FlashAttention 2
try:
    policy.pretrained_model.config.attn_implementation = "flash_attention_2"
except Exception:
    raise
optimizer = bnb.optim.Adam8bit(policy.parameters(), lr=ppo_cfg.learning_rate)

####### Remove value head and use same score for all tokens
from trl.core import (
    masked_whiten,
    masked_mean,
    masked_var,
    entropy_from_logits,
    flatten_dict,
)
class REINFORCETrainer(PPOTrainer):
    def compute_rewards(self, scores, logprobs, ref_logprobs, masks):
        rewards, non_score_rewards, kls = [], [], []
        for score, logprob, ref_logprob, mask in zip(scores, logprobs, ref_logprobs, masks):
            # no KL shaping at all
            kls.append(torch.zeros_like(logprob))
            non_score_reward = torch.zeros_like(logprob)

            reward = torch.zeros_like(logprob)
            reward[mask.bool()] = score  # SAME scalar on every response token
            rewards.append(reward)
            non_score_rewards.append(non_score_reward)

        return torch.stack(rewards), torch.stack(non_score_rewards), torch.stack(kls)

    def compute_advantages(self, values, rewards, mask):
        # No baseline, no discounting/smoothing, no whitening
        values = torch.zeros_like(rewards)
        advantages = rewards.detach() * mask
        returns = advantages
        return values, advantages, returns

    def loss(
        self,
        old_logprobs, values, logits, vpreds, logprobs, mask, advantages, returns
    ):
        # standard PPO clipping around the REINFORCE advantage
        ratio = torch.exp(logprobs - old_logprobs)
        pg_losses  = -advantages * ratio
        pg_losses2 = -advantages * torch.clamp(ratio,
                                               1.0 - self.config.cliprange,
                                               1.0 + self.config.cliprange)
        pg_loss = masked_mean(torch.max(pg_losses, pg_losses2), mask)

        # no value loss at all
        entropy = masked_mean(entropy_from_logits(logits), mask)

        stats = dict(
            loss=dict(policy=pg_loss.detach(), value=torch.tensor(0.0, device=pg_loss.device), total=pg_loss.detach()),
            policy=dict(
                entropy=entropy.detach(),
                approxkl=0.5 * masked_mean((logprobs - old_logprobs)**2, mask).detach(),
                policykl=masked_mean(old_logprobs - logprobs, mask).detach(),
                clipfrac=masked_mean((pg_losses2 > pg_losses).float(), mask).detach(),
                advantages=advantages.detach(),
                advantages_mean=masked_mean(advantages, mask).detach(),
                ratio=ratio.detach(),
            ),
            returns=dict(mean=masked_mean(returns, mask).detach(), var=masked_var(returns, mask).detach()),
            val=dict(vpred=torch.tensor(0.0, device=pg_loss.device), error=torch.tensor(0.0, device=pg_loss.device),
                     clipfrac=torch.tensor(0.0, device=pg_loss.device),
                     mean=torch.tensor(0.0, device=pg_loss.device), var=torch.tensor(0.0, device=pg_loss.device)),
        )
        # Return (policy loss, 0 * value loss)
        return pg_loss, torch.tensor(0.0, device=pg_loss.device), flatten_dict(stats)

ppo:PPOTrainer = [PPOTrainer,REINFORCETrainer][REINFORCE_STYLE](config=ppo_cfg, model=policy, tokenizer=tokenizer, optimizer=optimizer)

# Where we write the up-to-date LoRA adapter so vLLM can load it
ADAPTER_DIR = tempfile.mkdtemp(prefix="ppo_lora_")
# Permanent adapter directory (saved less frequently)
PERM_ADAPTER_DIR = os.path.join(os.path.dirname(__file__), "ppo_lora_adapter")
os.makedirs(PERM_ADAPTER_DIR, exist_ok=True)

ppo_update_count = 0  # counts successful PPO updates

# remove settings usage in log_event
def log_event(event: dict):
    event_with_time = {"ts": time.time(), "time":datetime.datetime.now().strftime('%Y%m%d-%H%M%S'),**event}
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
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
    return r.json() if r.headers.get("content-type","" ).startswith("application/json") else r.text

def unload_lora_adapter(lora_name: str):
    try:
        return _post("/v1/unload_lora_adapter", {"lora_name": lora_name})
    except Exception as e:
        # ignore if not loaded
        return {"status": "unloaded_or_missing"}

def load_lora_adapter(lora_name: str, lora_path: str):
    try:
        return _post("/v1/load_lora_adapter", {"lora_name": lora_name, "lora_path": lora_path})
    except RuntimeError as e:
        if 'has already been loaded' in str(e): # happens when old run loaded an adapter, and new run begins without restarting vllm server
            unload_lora_adapter(lora_name)
            return load_lora_adapter(lora_name, lora_path)

def _set_current_adapter(name: str):
    global CURRENT_ADAPTER_NAME
    CURRENT_ADAPTER_NAME = name

def _maybe_unload_old_adapters():
    # Keep only the most recent KEEP_ACTIVE_ADAPTERS adapters loaded.
    while len(_loaded_adapter_names) > KEEP_ACTIVE_ADAPTERS:
        old = _loaded_adapter_names.popleft()
        if old != CURRENT_ADAPTER_NAME:
            try:
                unload_lora_adapter(old)
            except Exception:
                pass

def load_new_adapter_version():
    """
    Save current PEFT weights, load them into vLLM under a versioned name based on ppo_update_count,
    and atomically switch NEW requests to the new name. Old adapters remain loaded briefly.
    """
    # Save PEFT adapter weights so vLLM can load them
    ppo.model.pretrained_model.save_pretrained(ADAPTER_DIR)
    new_name = f"ppo_adapter_{ppo_update_count}"
    load_lora_adapter(new_name, ADAPTER_DIR)
    _loaded_adapter_names.append(new_name)
    _set_current_adapter(new_name)
    _maybe_unload_old_adapters()

def initial_load_adapter():
    """
    Ensure an adapter is present before workers start issuing requests.
    Uses the current ppo_update_count (typically 0 at startup).
    """
    ppo.model.pretrained_model.save_pretrained(ADAPTER_DIR)
    initial_name = f"ppo_adapter_{ppo_update_count}"
    load_lora_adapter(initial_name, ADAPTER_DIR)
    _loaded_adapter_names.append(initial_name)
    _set_current_adapter(initial_name)

# Ensure adapter is present at startup so first generation does not 404
initial_load_adapter()

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
    if len(tokenizer(prompt, add_special_tokens=False).input_ids) > PROMPT_MAX_TOKENS:
        ctx.last_response_text = "[Error in generation]"
        return ctx

    try:
        resp = await aclient.completions.create(
            model=CURRENT_ADAPTER_NAME,
            prompt=prompt,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            extra_body=SAMPLING_EXTRA
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

def extract_move(text: str) -> Optional[Any]:
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

def messages_for_prompt_A(MOVES: int):
    sys = {"role": "system", "content": "You are a helpful assistant."}
    user = {"role": "user", "content": PROMPT_A_TMPL.format(BOARD_GAME_CODE=BOARD_GAME_CODE)}
    return [sys, user]

def msg_B(game_code: str, game_obj, remaining_moves: int, first: bool):
    legal = game_obj.get_legal_moves()
    if len(legal)>50:
        legal=legal[:40]+['... (omitted. deduct them by yourself) ...']+legal[-10:]
    board_str = repr(game_obj.board)
    if first:
        txt = PROMPT_B_FIRST_TMPL.format(GAME_CODE=game_code, remaining_moves=remaining_moves, legal=legal, board=board_str)
    else:
        txt = PROMPT_B_CONT_TMPL.format(remaining_moves=remaining_moves, legal=legal, board=board_str)
    return {"role": "user", "content": txt}

def msg_C(game_code: str, game_obj, goal_board, remaining_moves: int, first: bool):
    legal = game_obj.get_legal_moves()
    if len(legal)>50:
        legal=legal[:40]+['... (omitted. deduct them by yourself) ...']+legal[-10:]
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
    # direct links to per-goal PhaseBContext (next one will overwrite earlier)
    children: List['PhaseBContext'] = None
    last_prompt_text: str = ''
    last_response_text: str = ''

    def maybe_finalize_reward(self, finalize_reward):
        """
        Compute Reward-A once all B children have a success_rate.
        Reward-A = 10 * Var(p_i) across goals.
        """
        if not self.children or len(self.children) < self.N_goals:
            return
        ps = []
        for gi in range(self.N_goals):
            bctx = self.children[gi]
            if bctx is None or bctx.success_rate is None:
                return  # not ready
            ps.append(float(bctx.success_rate))

        # All p_i available → finalize Reward-A
        var_p = float(np.var(np.array(ps, dtype=float), ddof=0))
        if self.reward_idx is not None:
            finalize_reward([self.reward_idx], var_p * 10.0, Phase.A)

        # Build log record from B & C children state (no trackers needed)
        goals_payload = []
        for gi in range(self.N_goals):
            bctx = self.children[gi]
            tries_payload = []
            for cctx in (bctx.children or []):
                tries_payload.append({
                    "moves": list(cctx.moves_trace or []),
                    "final_board": repr(cctx.solver_game.board)
                })
            goals_payload.append({
                "goal_index": gi,
                "moves": list(bctx.moves_trace or []),
                "final_board": repr(bctx.proposer_game.board),
                "reward_B": bctx.reward_B if bctx.reward_B is not None else 0.0,
                "tries": tries_payload
            })

        record = {
            "game_code": self.game_code,
            "reward_A": var_p * 10.0,
            "goals": goals_payload
        }
        os.makedirs("logs", exist_ok=True)
        with open(GAME_COMPLETE_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def handle(self, enqueue, add_sample, finalize_reward):
        if self.children is None:
            self.children = []

        # One sample for Phase A (prompt/response pair)
        self.reward_idx = add_sample(self.last_prompt_text, self.last_response_text, None)
        
        # Filter sanity checks
        filterResult = filterGame(self.last_response_text)
        if filterResult != FilterGameResult.PASS:
            finalize_reward([self.reward_idx], filterResult.value, Phase.A, filterResult.name)
            return

        # Extract & compile game
        self.game_code = extract_game(self.last_response_text)
        self.GameClass = run_game_code_and_get_class(self.game_code) if self.game_code else None

        # Spawn one PhaseB per goal
        for gi in range(self.N_goals):
            proposer_game = self.GameClass()
            first_msg = msg_B(self.game_code, proposer_game, remaining_moves=self.K_moves, first=True)
            messages = [{"role": "system", "content": "You are a helpful assistant."}, first_msg]
            bctx = PhaseBContext(
                phase='B', messages=messages, thinking=self.thinking, GameClass=self.GameClass,
                proposer_game=proposer_game, moves_left=self.K_moves, goal_index=gi, parent_A=self,
                b_sample_indices=[], first_turn=True, N_tries=self.N_tries
            )
            # link initial B (will be overwritten by final B upon completion)
            self.children.append(bctx)
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
    # track chosen moves for this goal while proposing
    moves_trace: List[Any] = None
    # direct links to PhaseCContext roots (one per try)
    children: List['PhaseCContext'] = None

    # computed outcomes
    success_rate: Optional[float] = None
    reward_B: Optional[float] = None
    
    def _finalize_B(self, p: float, finalize_reward, note: Optional[str] = None, force_reward: float = None):
        """Idempotently set success_rate and Reward-B, then finalize."""
        if self.success_rate is None:
            self.success_rate = float(p)
        if self.reward_B is None:
            r_b = max(0.0, 1.0 - self.success_rate)
            if self.success_rate == 0.0:
                r_b = 0.0
            if force_reward is not None:
                r_b = force_reward
            self.reward_B = r_b
            if self.b_sample_indices:
                note = note or f"p={self.success_rate:.3f}"
                finalize_reward(self.b_sample_indices, r_b, Phase.B, note=note)
        # ask parent A to try finalizing Reward-A
        self.parent_A.maybe_finalize_reward(finalize_reward)

    def maybe_finalize_reward(self, finalize_reward):
        """
        If all C tries (children) have completed, compute p = successes / tries,
        finalize Reward-B for all B samples, and then ask A to maybe finalize.
        """
        # If p already known (e.g., short-circuit case), ensure reward persisted:
        if self.success_rate is not None:
            self._finalize_B(self.success_rate, finalize_reward)
            return

        # Otherwise wait for all C tries
        if not self.children or len(self.children) < self.N_tries:
            return
        if any(not getattr(c, "is_complete", False) for c in self.children):
            return

        successes = sum(1 for c in self.children if bool(c.success))
        tries = len(self.children)
        p = successes / float(tries) if tries > 0 else 0.0
        # for c in self.children:
        #     reward_c = (1.0 if c.success else 0.0) - p
        #     if c.try_sample_indices:
        #         finalize_reward(c.try_sample_indices, reward_c, Phase.C,
        #                         note=f"{'Success' if c.success else 'Fail'} (p={p:.3f})")
        self._finalize_B(p, finalize_reward)

    def handle(self, enqueue, add_sample, finalize_reward):
        if self.moves_trace is None:
            self.moves_trace = []
        if self.children is None:
            self.children = []
        # Link self to parent A (overwrites earlier if multi-turn)
        self.parent_A.children[self.goal_index] = self

        # Log B turn sample
        idx = add_sample(self.last_prompt_text, self.last_response_text, None)
        self.b_sample_indices.append(idx)

        # Choose & validate move
        raw_chosen = chosen = extract_move(self.last_response_text)
        legal = self.proposer_game.get_legal_moves()
        is_valid = chosen == 'DONE' or (chosen is not None and (chosen in legal or str(chosen) in legal))

        if not is_valid:
            # terminate this goal construction attempt to avoid infinite loops
            chosen = 'DONE'

        if chosen is not None and chosen != 'DONE':
            # record and execute the valid move
            self.moves_trace.append(chosen)
            self.proposer_game.execute_move(chosen)
            self.moves_left -= 1

        # Continue proposing if moves remain and not DONE
        if self.moves_left > 0 and chosen != 'DONE':
            new_messages = self.messages + [
                {"role": "assistant", "content": remove_think(self.last_response_text)},
                msg_B(self.parent_A.game_code, self.proposer_game, remaining_moves=self.moves_left, first=False)
            ]
            enqueue(PhaseBContext(
                phase='B', messages=new_messages, thinking=self.thinking, GameClass=self.GameClass,
                proposer_game=self.proposer_game, moves_left=self.moves_left, goal_index=self.goal_index,
                parent_A=self.parent_A, b_sample_indices=self.b_sample_indices, first_turn=False, N_tries=self.N_tries,
                moves_trace=self.moves_trace, children=self.children
            ))
            return

        # Terminal conditions
        if self.first_turn and (chosen == 'DONE' or self.moves_left == self.parent_A.K_moves):
            # C always solves in this trivial case -> p = 1.0, Reward-B = 0
            self._finalize_B(1.0, finalize_reward, note=f"p=1.000 (DONE on first turn). raw chosen: {raw_chosen}, legal moves: {legal}", force_reward=-1)
            return

        # Otherwise, finalize this goal state and spawn C tries
        goal_board = copy.deepcopy(self.proposer_game.board)
        for ti in range(self.N_tries):
            solver_game = self.GameClass()
            c_first_msg = msg_C(self.parent_A.game_code, solver_game, goal_board,
                                remaining_moves=self.parent_A.K_moves, first=True)
            messages = [{"role": "system", "content": "You are a helpful assistant."}, c_first_msg]
            cctx = PhaseCContext(
                phase='C', messages=messages, thinking=self.thinking, GameClass=self.GameClass,
                solver_game=solver_game, goal_board=goal_board, moves_left=self.parent_A.K_moves,
                goal_index=self.goal_index, try_index=ti, parent_A=self.parent_A, parent_B=self,
                try_sample_indices=[]
            )
            self.children.append(cctx)
            enqueue(cctx)


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
    try_index: int
    parent_A: PhaseAContext
    parent_B: PhaseBContext
    try_sample_indices: List[int]
    last_prompt_text: str = ''
    last_response_text: str = ''
    # track chosen moves for this solver try
    moves_trace: List[Any] = None

    # computed outcome for this try
    success: Optional[bool] = None
    is_complete: bool = False

    def handle(self, enqueue, add_sample, finalize_reward):
        if self.moves_trace is None:
            self.moves_trace = []
            
        self.parent_B.children[self.try_index] = self

        # Record this C step
        sample_idx = add_sample(self.last_prompt_text, self.last_response_text, None)
        self.try_sample_indices.append(sample_idx)

        # Extract and validate move
        chosen = extract_move(self.last_response_text)
        legal = self.solver_game.get_legal_moves()
        valid = (chosen is not None) and (chosen in legal or str(chosen) in legal)

        if valid:
            self.moves_trace.append(chosen)
            self.solver_game.execute_move(chosen)
            self.moves_left -= 1
        else:
            # invalid move or no move: end this try
            self.moves_left = 0

        # Keep solving if moves remain and goal not yet matched
        if self.moves_left > 0 and self.solver_game.board != self.goal_board:
            new_messages = self.messages + [
                {"role": "assistant", "content": remove_think(self.last_response_text)},
                msg_C(self.parent_A.game_code, self.solver_game, self.goal_board,
                      remaining_moves=self.moves_left, first=False)
            ]
            enqueue(PhaseCContext(
                phase='C', messages=new_messages, thinking=self.thinking, GameClass=self.GameClass,
                solver_game=self.solver_game, goal_board=self.goal_board, moves_left=self.moves_left,
                goal_index=self.goal_index, try_index=self.try_index, parent_A=self.parent_A, parent_B=self.parent_B,
                try_sample_indices=self.try_sample_indices, moves_trace=self.moves_trace
            ))
            return

        # Try completed; compute Reward-C for this try
        self.success = (self.solver_game.board == self.goal_board)
        self.is_complete = True
        r_try = 1.0 if self.success else 0.0
        if self.try_sample_indices:
            finalize_reward(self.try_sample_indices, r_try, Phase.C)

        # Let Phase B see if all tries are done so it can compute Reward-B
        self.parent_B.maybe_finalize_reward(finalize_reward)

# ----------------------------
# Evaluation Context (highest priority)
# ----------------------------
@dataclass
class EvalContext:
    phase: str  # 'EVAL'
    messages: List[Dict]
    thinking: bool
    eval_index: int
    sample_index: int
    game_code: str
    sequence: List[Any]
    final_board: Any
    GameClass: Any
    moves_left: int
    solver_game: Any
    model_moves: List[Any] = None
    last_prompt_text: str = ''
    last_response_text: str = ''
    is_complete: bool = False

    def handle(self, enqueue, add_sample, finalize_reward):
        global EVAL_RUNS
        if self.model_moves is None:
            self.model_moves = []
        add_sample(self.last_prompt_text, self.last_response_text, None)
        chosen = extract_move(self.last_response_text)
        legal = self.solver_game.get_legal_moves()
        valid = (chosen is not None) and (chosen in legal or str(chosen) in legal)
        if valid and self.moves_left > 0:
            self.model_moves.append(chosen)
            self.solver_game.execute_move(chosen)
            self.moves_left -= 1
        else:
            self.moves_left = 0
        if self.moves_left > 0 and self.solver_game.board != self.final_board:
            new_messages = self.messages + [
                {"role": "assistant", "content": remove_think(self.last_response_text)},
                msg_C(self.game_code, self.solver_game, self.final_board, remaining_moves=self.moves_left, first=False)
            ]
            enqueue(EvalContext(
                phase='EVAL', messages=new_messages, thinking=self.thinking, eval_index=self.eval_index,
                sample_index=self.sample_index, game_code=self.game_code, sequence=self.sequence,
                final_board=self.final_board, GameClass=self.GameClass, moves_left=self.moves_left,
                solver_game=self.solver_game, model_moves=self.model_moves
            ))
            return
        success = (self.solver_game.board == self.final_board)
        run = EVAL_RUNS[self.eval_index]
        run['results'][self.sample_index] = {
            'sample_index': self.sample_index,
            'game': self.game_code,
            'sequence': self.sequence,
            'final_board': self.final_board,
            'model_moves': self.model_moves,
            'model_final_board': self.solver_game.board,
            'success': success
        }
        if all(r is not None for r in run['results']):
            samples = run['results']
            overall_acc = float(sum(1 for r in samples if r['success']) / len(samples)) if samples else 0.0
            per_game_counts = {}
            per_game_success = {}
            for r in samples:
                g = r['game']
                per_game_counts[g] = per_game_counts.get(g, 0) + 1
                if r['success']:
                    per_game_success[g] = per_game_success.get(g, 0) + 1
            per_game_acc = {g: per_game_success.get(g, 0)/c for g, c in per_game_counts.items()}
            run['summary'] = {'overall_acc': overall_acc, 'per_game_acc': per_game_acc}
            os.makedirs(os.path.dirname(EVAL_RESULTS_FILE), exist_ok=True)
            with open(EVAL_RESULTS_FILE, 'a', encoding='utf-8') as f:
                f.write(json.dumps(run, ensure_ascii=False) + '\n')
            log_event({'phase': 'EVAL', 'eval_index': self.eval_index, 'summary': run['summary']})
        self.is_complete = True

# ----------------------------
# Queue-based infinite PPO loop (removed one_ppo_step)
# ----------------------------

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

def finalize_reward(idx_list: List[int], value: float, phase: Phase, note: Optional[str] = ''):
    t = torch.tensor(float(value), device=ppo.accelerator.device)
    for idx in idx_list:
        if rewards[idx] is None:
            rewards[idx] = t
            unsent_indices.append(idx)
            log_event({"prompt_text": query_texts[idx], "response_text": response_texts[idx], "phase": phase.value, "reward": float(value), "note": note})

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
            isCropped=False
            if query_token_count + resp_token_count <= PPO_MAX_TOKENS:
                filtered.append((qi, ri, rw, idx))
            elif query_token_count < PPO_MAX_TOKENS:
                isCropped=True
                # Crop response to fit. reward -= 1
                allowed_resp_tokens = PPO_MAX_TOKENS - query_token_count
                ri_cropped = ri[:allowed_resp_tokens]
                filtered.append((qi, ri_cropped, rw-1, idx))
            else:
                isRemoved=True
                removed.append({"idx": idx, "query_tokens": int(query_token_count), "resp_tokens": int(resp_token_count)})
            log_event({
                "phase": "FILTER", 
                "query_text": qt,  
                "resp_text": rt,    
                "actual_query_tokens": int(query_token_count),
                "actual_resp_tokens": int(resp_token_count),
                'sum_tokens': int(query_token_count + resp_token_count),
                "isCropped": isCropped,
                "isRemoved": isRemoved,
            })
        if not filtered:
            return
        query_ids, resp_ids, batch_rewards, batch_indices = zip(*filtered)
        stats = ppo.step(list(query_ids), list(resp_ids), list(batch_rewards))
        log_event({"phase": "PPO", "stats": {k: (float(v) if isinstance(v, (int, float)) else str(v)) for k, v in stats.items()}, "num_samples": len(filtered)})
        # Permanent save every settings.perm_save_interval updates
        global ppo_update_count
        ppo_update_count += 1
        if ppo_update_count % 32 == 0:
            # Load a NEW versioned adapter and switch new requests to it.
            # Older adapters remain temporarily loaded to avoid races.
            load_new_adapter_version()
        if ppo_update_count % PERM_SAVE_INTERVAL == 0:
            path=PERM_ADAPTER_DIR+'/'+START_TIME_STR+f'/{ppo_update_count}'
            ppo.model.pretrained_model.save_pretrained(path)
            log_event({"phase": "PERM_SAVE", "update": ppo_update_count, "path": path})
            # trigger evaluation if period reached
            if (ppo_update_count % (PERM_SAVE_INTERVAL * EVAL_PERIOD) == 0):
                    trigger_evaluation()


def trigger_evaluation():
    samples = load_test_samples()
    if not samples:
        print("No evaluation samples found.")
        return
    global CURRENT_EVAL_INDEX, EVAL_RUNS
    CURRENT_EVAL_INDEX += 1
    results_placeholder = [None]*len(samples)
    EVAL_RUNS.append({'index': CURRENT_EVAL_INDEX, 'summary': {}, 'results': results_placeholder})
    # enqueue eval contexts
    for si, s in enumerate(samples):
        GameClass = run_game_code_and_get_class(s.gameCode)
        solver_game = GameClass()
        first_msg = msg_C(s.gameCode, solver_game, s.finalBoard, remaining_moves=K_MOVES, first=True)
        eval_ctx = EvalContext(
            phase='EVAL', messages=[{"role": "system", "content": "You are a helpful assistant."}, first_msg],
            thinking=THINKING, eval_index=CURRENT_EVAL_INDEX, sample_index=si,
            game_code=s.gameCode, sequence=s.sequence, final_board=s.finalBoard,
            GameClass=GameClass, moves_left=K_MOVES, solver_game=solver_game
        )
        PENDING_EVAL_CONTEXTS.append(eval_ctx)

# Global evaluation accumulator (moved earlier to avoid forward reference issues)
EVAL_RUNS: list = []  # list of dict per evaluation index
CURRENT_EVAL_INDEX: int = -1
PENDING_EVAL_CONTEXTS: List[Any] = []
TEST_SAMPLES = None

def load_test_samples():
    global TEST_SAMPLES
    if TEST_SAMPLES is None:
        try:
            from evalGames import testSamples
            TEST_SAMPLES = testSamples
        except Exception:
            TEST_SAMPLES = []
    return TEST_SAMPLES

if EVAL_AT_INIT:
    trigger_evaluation()  # initial evaluation at startup

# ----------------------------
# Priority queue + worker setup
# ----------------------------

# Priorities: C first, then B, then A (like your heap)
PHASE_PRIO = {'EVAL': -1, 'C': 0, 'B': 1, 'A': 2}

async def run_workers_and_feed():
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
            messages=messages_for_prompt_A(MOVES=K_MOVES),
            K_moves=K_MOVES,
            N_goals=N_GOALS,
            N_tries=N_TRIES,
            thinking=THINKING,
        )
        active_As.append(a)
        await enqueue(a)

    # keep the queue topped up so workers never idle
    async def feeder():
        while True:
            # push pending eval contexts first
            while PENDING_EVAL_CONTEXTS:
                ctx = PENDING_EVAL_CONTEXTS.pop(0)
                await enqueue(ctx)
            # If low backlog, add more A to spawn B/C work later
            if pq.qsize() < BATCH_GEN_LIMIT // 4:
                await new_phase_A()
            await asyncio.sleep(0.005)

    # Worker: always one prompt per HTTP request
    ppo_lock = asyncio.Lock()  # serialize PPO updates
    async def worker(wid: int):
        while True:
            _, _, ctx = await pq.get()
            try:
                await _gen_one(ctx, thinking=THINKING)
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
                with open(GENERATION_LOG,'a',encoding='utf-8') as f:
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
    await run_workers_and_feed()

if __name__ == "__main__":
    asyncio.run(main())