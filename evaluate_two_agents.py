import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import logging
import argparse
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from config import Config, setup_logging
from games import TicTacToeGame, Connect4Game
from agents import RandomAgent, MinimaxAgent, APIAgent, VLLMAgent
from evaluation.evaluator import Evaluator
from utils import create_agent

try:
    from openai import OpenAI
    import httpx
except ImportError:
    print("WARNING: OpenAI Python package not installed. Install with 'pip install openai httpx'")
    OpenAI = None
    httpx = None

def parse_args():
    parser = argparse.ArgumentParser(description="Two Agent Battle Evaluation for TicTacToe")
    parser.add_argument("--num_games", type=int, default=50, help="Number of games to evaluate")
    
    # Agent 1 configuration
    parser.add_argument("--agent1", type=str, required=True, 
                        choices=["api", "random", "minimax", "vllm"],
                        help="First agent type")
    parser.add_argument("--agent1_model", type=str, default="gpt-4-0125-preview", 
                        help="Model name for agent1 (only used if agent1 is 'api')")
    parser.add_argument("--agent1_api_base_url", type=str, 
                        help="API base URL for agent1 (required if agent1 is 'api')")
    parser.add_argument("--agent1_api_key", type=str, 
                        help="API key for agent1 (required if agent1 is 'api')")
    parser.add_argument("--agent1_model_path", type=str, 
                        default='Qwen/Qwen3-8B',
                        help="Model path for agent1 (only used if agent1 is 'vllm')")
    
    # Agent 2 configuration
    parser.add_argument("--agent2", type=str, required=True, 
                        choices=["api", "random", "minimax", "vllm"],
                        help="Second agent type")
    parser.add_argument("--agent2_model", type=str, default="gpt-4-0125-preview", 
                        help="Model name for agent2 (only used if agent2 is 'api')")
    parser.add_argument("--agent2_api_base_url", type=str, 
                        help="API base URL for agent2 (required if agent2 is 'api')")
    parser.add_argument("--agent2_api_key", type=str, 
                        help="API key for agent2 (required if agent2 is 'api')")
    parser.add_argument("--agent2_model_path", type=str, 
                        default='Qwen/Qwen3-8B',
                        help="Model path for agent2 (only used if agent2 is 'vllm')")
    
    parser.add_argument("--output_dir", type=str, help="Output directory for results")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Initialize config
    config = Config()
    
    # Override config with command line arguments
    if args.num_games:
        config.NUM_EVAL_GAMES = args.num_games
    if args.output_dir:
        config.OUTPUT_DIR_BASE = args.output_dir
    
    # Setup logging
    logger = setup_logging(config)
    logger.info(f"Starting two-agent battle evaluation: {args.agent1} vs {args.agent2}")
    
    # Create agent 1
    try:
        agent1 = create_agent(
            args.agent1, 
            f"Agent1-{args.agent1}",
            model=args.agent1_model,
            api_base_url=args.agent1_api_base_url,
            api_key=args.agent1_api_key,
            model_path=args.agent1_model_path
        )
        logger.info(f"Successfully initialized Agent1: {agent1.name}")
    except Exception as e:
        logger.critical(f"Failed to initialize Agent1 ({args.agent1}): {e}")
        return 1
    
    # Create agent 2
    try:
        agent2 = create_agent(
            args.agent2, 
            f"Agent2-{args.agent2}",
            model=args.agent2_model,
            api_base_url=args.agent2_api_base_url,
            api_key=args.agent2_api_key,
            model_path=args.agent2_model_path
        )
        logger.info(f"Successfully initialized Agent2: {agent2.name}")
    except Exception as e:
        logger.critical(f"Failed to initialize Agent2 ({args.agent2}): {e}")
        return 1
    
    # Initialize evaluator
    evaluator = Evaluator(config)
    
    # Run evaluation
    logger.info(f"Starting battle: {agent1.name} vs {agent2.name}")
    results = evaluator.evaluate_agent_vs_agent(
        agent1, agent2, Connect4Game, config.NUM_EVAL_GAMES
    )
    
    # Save results
    evaluator.save_results([results])
    
    # Print summary
    logger.info("=" * 50)
    logger.info("BATTLE RESULTS SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Agent1 ({agent1.name}) vs Agent2 ({agent2.name})")
    logger.info(f"Total games: {config.NUM_EVAL_GAMES}")
    logger.info(f"Results: {results}")
    logger.info("=" * 50)
    logger.info("Two-agent battle evaluation complete!")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())