import os
import logging
import argparse
import sys

# Ensure CUDA_VISIBLE_DEVICES is set early
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from config import Config, setup_logging
from games.tictactoe import TicTacToeGame
from agents.random_agent import RandomAgent
from agents.minimax_agent import MinimaxAgent
from agents.vllm_agent import VLLMAgent
from evaluation.evaluator import Evaluator
from utils.model_utils import ModelUtils

def parse_args():
    parser = argparse.ArgumentParser(description="Chess Battle Assessment Framework")
    parser.add_argument("--model_path", type=str, help="Path to the model")
    parser.add_argument("--num_games", type=int, help="Number of games to evaluate")
    parser.add_argument("--temperature", type=float, help="Temperature for sampling")
    parser.add_argument("--top_p", type=float, help="Top-p for sampling")
    parser.add_argument("--parallel_size", type=int, help="Tensor parallel size for vLLM")
    parser.add_argument("--output_dir", type=str, help="Output directory for results")
    parser.add_argument("--eval_type", type=str, choices=["random", "minimax", "both"], 
                        default="both", help="Evaluation type")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Initialize config
    config = Config()
    
    # Override config with command line arguments
    if args.model_path:
        config.MODEL_PATH = args.model_path
    if args.num_games:
        config.NUM_EVAL_GAMES = args.num_games
    if args.temperature:
        config.TEMPERATURE = args.temperature
    if args.top_p:
        config.TOP_P = args.top_p
    if args.parallel_size:
        config.VLLM_TENSOR_PARALLEL_SIZE = args.parallel_size
    if args.output_dir:
        config.OUTPUT_DIR_BASE = args.output_dir
        
    # Setup logging
    logger = setup_logging(config)
    
    # Check if model path exists
    if not os.path.isdir(config.MODEL_PATH):
        logger.critical(f"Model path does not exist: {config.MODEL_PATH}")
        logger.critical("Please update 'MODEL_PATH' in the Config class or provide a valid path.")
        return 1
    
    # Initialize vLLM model
    try:
        llm_engine, sampling_params, tokenizer = ModelUtils.initialize_vllm_model(
            config.MODEL_PATH, config.VLLM_TENSOR_PARALLEL_SIZE
        )
        
        # Update sampling parameters
        sampling_params = ModelUtils.update_sampling_params(
            sampling_params,
            temperature=config.TEMPERATURE,
            top_p=config.TOP_P,
            max_tokens=config.MAX_GENERATION_LENGTH
        )
    except Exception as e:
        logger.critical(f"Failed to initialize vLLM model: {e}")
        return 1
    
    # Initialize agents
    llm_agent = VLLMAgent(llm_engine, sampling_params, tokenizer, name="LLMAgent")
    random_agent = RandomAgent(name="RandomAgent")
    minimax_agent = MinimaxAgent(name="MinimaxAgent")
    
    # Initialize evaluator
    evaluator = Evaluator(config)
    
    # Run evaluations
    results = []
    
    if args.eval_type in ["random", "both"]:
        logger.info("Starting evaluation against Random agent")
        random_results = evaluator.evaluate_agent_vs_agent(
            llm_agent, random_agent, TicTacToeGame, config.NUM_EVAL_GAMES
        )
        results.append(random_results)
    
    if args.eval_type in ["minimax", "both"]:
        logger.info("Starting evaluation against Minimax agent")
        minimax_results = evaluator.evaluate_agent_vs_agent(
            llm_agent, minimax_agent, TicTacToeGame, config.NUM_EVAL_GAMES
        )
        results.append(minimax_results)
    
    # Save results
    evaluator.save_results(results)
    
    logger.info("Evaluation complete!")
    return 0

if __name__ == "__main__":
    sys.exit(main())