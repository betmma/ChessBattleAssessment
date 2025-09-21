# settings.py
BASE_MODEL = "/remote-home1/share/models/Qwen3-8B"
VLLM_URL = "http://localhost:8000"
VLLM_KEY = "token-abc123"
KEEP_ACTIVE_ADAPTERS = 8
BATCH_GEN_LIMIT = 128
MAX_TOKENS = 24000
PROMPT_MAX_TOKENS = 7500 # MAX_TOKENS + PROMPT_MAX_TOKENS should be <= model len when running vllm server
PPO_MAX_TOKENS = 5500
TEMPERATURE = 0.6
TOP_P = 0.95
TOP_K = 20
PERM_SAVE_INTERVAL = 200
LOG_FILE = "logs/ppo_rollout_log.jsonl"
GAME_COMPLETE_LOG = "logs/gameComplete.log"
GENERATION_LOG = "logs/generations.log"

# Evaluation configuration (run evaluation every EVAL_PERIOD multiples of PERM_SAVE_INTERVAL)
EVAL_PERIOD = 4  # means every (EVAL_PERIOD * PERM_SAVE_INTERVAL) PPO updates trigger evaluation after save
EVAL_RESULTS_FILE = "logs/eval_results.jsonl"
EVAL_AT_INIT = False # whether to run eval at the very start before any training (though, async will cause some evaluations happen after some ppo updates)

# Run configuration constants
K_MOVES = 8
N_GOALS = 10
N_TRIES = 10
THINKING = True

# Derived values
SAMPLING_EXTRA = {"top_p": TOP_P, "top_k": TOP_K,}# "repetition_penalty": 1.005}

REINFORCE_STYLE = True # remove value head, use only policy head and REINFORCE style policy gradient
