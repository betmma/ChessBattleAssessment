import os
import logging
from datetime import datetime

class Config:
    # VLLM Settings
    VLLM_TENSOR_PARALLEL_SIZE = 1
    VLLM_MAX_NUM_SEQS = 4096
    VLLM_MAX_NUM_BATCHED_TOKENS = 16384
    VLLM_MAX_MODEL_LEN = 2048  # Max length of the model context
    VLLM_GPU_MEMORY_UTILIZATION = 0.9  # GPU memory utilization for vLLM
    
    MAX_CONCURRENT_GAMES = 1  # How many games to process in parallel logic stages
    
    # Evaluation Settings
    NUM_EVAL_GAMES = 50  # Default number of games to evaluate
    
    # Generation Settings
    MAX_PROMPT_LENGTH = 512  # Max length for tokenizer context for prompts
    MAX_GENERATION_LENGTH = 1024  # Max new tokens for LLM to generate for a move
    TEMPERATURE = 0.1
    TOP_P = 0.95
    LOCAL_ENABLE_THINKING = True  # Enable thinking for local agents
    
    # Logging Settings
    LOG_LEVEL = logging.INFO
    OUTPUT_DIR_BASE = "./evaluation_results"
    CUDA_VISIBLE_DEVICES = "7"
    LOG_ACTION_REWARDS = True  # Enable logging of action rewards during evaluation
    
    @property
    def OUTPUT_DIR(self):
        return f"{self.OUTPUT_DIR_BASE}_vllm"

def setup_logging(config=None):
    if config is None:
        config = Config()
        
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    log_file = os.path.join(config.OUTPUT_DIR, f"evaluation_log_{datetime.now().strftime('%Y%m%d-%H%M%S')}.log")
    
    logging.basicConfig(
        level=config.LOG_LEVEL,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logging.info(f"Logging to {log_file}")
    return logging.getLogger(__name__)