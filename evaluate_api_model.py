import os
import logging
import argparse
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from config import Config, setup_logging
from games.tictactoe import TicTacToeGame
from agents.random_agent import RandomAgent
from agents.minimax_agent import MinimaxAgent
from agents.api_agent import APIAgent
from evaluation.evaluator import Evaluator

try:
    from openai import OpenAI
    import httpx
except ImportError:
    print("WARNING: OpenAI Python package not installed. Install with 'pip install openai httpx'")
    raise

def parse_args():
    parser = argparse.ArgumentParser(description="API Model Evaluation for TicTacToe")
    parser.add_argument("--num_games", type=int, default=50, help="Number of games to evaluate")
    parser.add_argument("--model", type=str, default="gpt-4-0125-preview", help="API model to use")
    parser.add_argument("--api_base_url", type=str, required=True, help="API base URL")
    parser.add_argument("--api_key", type=str, required=True, help="API key")
    parser.add_argument("--output_dir", type=str, help="Output directory for results")
    parser.add_argument("--opponent", type=str, choices=["random", "minimax", "both"], 
                        default="both", help="Opponent type to evaluate against")
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
    logger.info(f"Starting API model evaluation using {args.model}")
    
    # Initialize API client
    try:
        client = OpenAI(
            base_url=args.api_base_url,
            api_key=args.api_key,
            http_client=httpx.Client(
                base_url=args.api_base_url,
                follow_redirects=True,
            ),
            timeout=httpx.Timeout(600, read=600, write=600, connect=600),
        )
    except Exception as e:
        logger.critical(f"Failed to initialize API client: {e}")
        return 1
    
    # Initialize agents
    api_agent = APIAgent(
        api_client=client, 
        model=args.model,
        name=f"APIAgent-{args.model}"
    )
    random_agent = RandomAgent(name="RandomAgent")
    minimax_agent = MinimaxAgent(name="MinimaxAgent")
    
    # Initialize evaluator
    evaluator = Evaluator(config)
    
    # Run evaluations
    results = []
    
    if args.opponent in ["random", "both"]:
        logger.info("Starting evaluation against Random agent")
        random_results = evaluator.evaluate_agent_vs_agent(
            api_agent, random_agent, TicTacToeGame, config.NUM_EVAL_GAMES
        )
        results.append(random_results)
    
    if args.opponent in ["minimax", "both"]:
        logger.info("Starting evaluation against Minimax agent")
        minimax_results = evaluator.evaluate_agent_vs_agent(
            api_agent, minimax_agent, TicTacToeGame, config.NUM_EVAL_GAMES
        )
        results.append(minimax_results)
    
    # Save results
    evaluator.save_results(results)
    
    logger.info("API model evaluation complete!")
    return 0

if __name__ == "__main__":
    sys.exit(main())