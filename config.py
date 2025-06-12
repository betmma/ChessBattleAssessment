import os
import logging
from datetime import datetime

class Config:
    # Model Settings
    MODEL_PATH = "/mnt/data/user/zhao_jun/mou_yu_rong/openrlhf/chessBattleAdvanced/Qwen3-8B"
    
    # VLLM Settings
    VLLM_TENSOR_PARALLEL_SIZE = 1
    VLLM_MAX_CONCURRENT_GAMES = 500  # How many games to process in parallel logic stages
    
    # Generation Settings
    MAX_PROMPT_LENGTH = 512  # Max length for tokenizer context for prompts
    MAX_GENERATION_LENGTH = 256  # Max new tokens for LLM to generate for a move
    DO_SAMPLE_GENERATION = True
    TEMPERATURE = 0.1
    TOP_P = 0.95
    
    # Evaluation Settings
    NUM_EVAL_GAMES = 500  # Number of games per opponent type
    
    # Logging Settings
    LOG_LEVEL = logging.INFO
    OUTPUT_DIR_BASE = "./evaluation_results"
    CUDA_VISIBLE_DEVICES = "7"
    
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