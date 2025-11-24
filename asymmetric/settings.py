# settings.py
BASE_MODEL = "/remote-home1/share/models/Qwen3-8B"
VLLM_URL = "http://localhost:8000"
VLLM_KEY = "token-abc123"
KEEP_ACTIVE_ADAPTERS = 16
LORA_RANK = 16
BATCH_GEN_LIMIT = 64
MAX_TOKENS = 8000
PROMPT_MAX_TOKENS = 7500 # MAX_TOKENS + PROMPT_MAX_TOKENS should be <= model len when running vllm server
PPO_MAX_TOKENS = 5500
TEMPERATURE = 0.6
TOP_P = 0.95
TOP_K = 20
LEARNING_RATE = 1e-5
BATCH_SIZE = 64 # equals to gradient_accumulation_steps. mini_batch_size is always 1
PERM_SAVE_INTERVAL = 5 # save every N PPO updates
LOG_FILE = "ppo_rollout_log.jsonl"
GAME_COMPLETE_LOG = "gameComplete.log"
GENERATION_LOG = "generations.log"

# Evaluation configuration (run evaluation every EVAL_PERIOD multiples of PERM_SAVE_INTERVAL)
EVAL_PERIOD = 4  # means every (EVAL_PERIOD * PERM_SAVE_INTERVAL) PPO updates trigger evaluation after save
EVAL_RESULTS_FILE = "eval_results.jsonl"
EVAL_AT_INIT = False # whether to run eval at the very start before any training (though, async will cause some evaluations happen after some ppo updates)

# Run configuration constants
K_MOVES = 8
N_GOALS = 10
N_TRIES = 10
THINKING = True

# Derived values
SAMPLING_EXTRA = {"top_p": TOP_P, "top_k": TOP_K,}# "repetition_penalty": 1.005}

REINFORCE_STYLE = False # remove value head, use only policy head and REINFORCE style policy gradient
SYNC_REWARDS = True # whether to sync N_TRIES phase C rewards to appear only if all N_TRIES phase C finished (if false, early stopped games will send lower rewards before full length game rewards, and can cause uneven distribution), and deduct it by p(success), imitating grpo. same to phase Bs
DONT_TRAIN_PHASE_A = True # phase A reward is much less (K_MOVES * N_GOALS * N_TRIES times than phase C, except for failed games) but causes many negative rewards in the beginning (failed games)
USE_FIXED_GAMES = True # use 5 evalGames instead of let model generate (remove phase A). DONT_TRAIN_PHASE_A should be True if this is True.

# it's normal to see one-step problems coming first, since one-step problems end faster than multi-step problems and are sent to phase C earlier.