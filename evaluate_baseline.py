import os
import logging
import argparse
import sys
import time
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from config import Config, setup_logging
from games.tictactoe import TicTacToeGame
from agents.random_agent import RandomAgent
from agents.minimax_agent import MinimaxAgent
from evaluation.evaluator import Evaluator

def parse_args():
    parser = argparse.ArgumentParser(description="Baseline Evaluation of Random vs Minimax Agents")
    parser.add_argument("--num_games", type=int, default=500, help="Number of games to evaluate")
    parser.add_argument("--output_dir", type=str, help="Output directory for results")
    return parser.parse_args()

def main():
    # Parse command line arguments
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
    logger.info("Starting baseline evaluation: Random vs Minimax")
    
    # Initialize agents
    random_agent = RandomAgent(name="RandomAgent")
    minimax_agent = MinimaxAgent(name="MinimaxAgent")
    
    # Initialize evaluator
    evaluator = Evaluator(config)
    
    # Run Random vs Minimax evaluation
    start_time = time.time()
    results = evaluator.evaluate_agent_vs_agent(
        random_agent, minimax_agent, TicTacToeGame, config.NUM_EVAL_GAMES
    )
    end_time = time.time()
    
    # Log results
    logger.info(f"Evaluation completed in {end_time - start_time:.2f} seconds")
    logger.info(f"Random vs Minimax Results: {results}")
    
    # For better understanding, let's log a summary
    total_completed = results['total_games'] - results['forfeits_agent1'] - results['forfeits_agent2']
    logger.info(f"Random Wins: {results['wins_agent1']} ({results['win_rate_agent1']:.2%})")
    logger.info(f"Minimax Wins: {results['wins_agent2']} ({results['win_rate_agent2']:.2%})")
    logger.info(f"Draws: {results['draws']} ({results['draw_rate']:.2%})")
    
    # Save results
    evaluator.save_results([results])
    
    logger.info("Baseline evaluation complete!")
    return 0

if __name__ == "__main__":
    sys.exit(main())