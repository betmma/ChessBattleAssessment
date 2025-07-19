"""
Comprehensive BoardGame evaluation program to assess:
1. Balance: first player win rate - second player win rate
2. Deterministic: non-draw rate
3. Length: average steps per game
4. Controllability: average legal moves per state
5. Strategy depth: minimax depth comparisons
6. Variation: minimax with temperature analysis
"""

import os
import sys
import logging
import time
import statistics
import json
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import Config, setup_logging
from games import Games, GameByName
from agents import RandomAgent, MinimaxAgent
from evaluation.evaluator import Evaluator
from games.board_game import BoardGame


class BoardGameBalanceEvaluator:
    """Comprehensive evaluator for BoardGame balance metrics"""
    
    def __init__(self, config: Config = None, num_games: int = 50):
        self.config = config or Config()
        self.num_games = num_games
        self.evaluator = Evaluator(config=self.config, retry_limit=3)
        self.results = {}
        
        # Setup logging
        self.logger = setup_logging(self.config)
        
        # Cache for game logs to avoid re-running evaluations
        self.game_logs_cache = {}
    
    def get_or_cache_game_logs(self, game_class: type, agent1, agent2, cache_key: str) -> Tuple[Dict, Any]:
        """Get game logs from cache or run evaluation"""
        if cache_key in self.game_logs_cache:
            return self.game_logs_cache[cache_key]
        
        results, logger = self.evaluator.evaluate_agent_vs_agent_with_logger(
            agent1, agent2, game_class, self.num_games
        )
        
        self.game_logs_cache[cache_key] = (results, logger)
        return results, logger
        
    def debug_game_logs(self, game_class: type, agent1, agent2) -> None:
        """Debug method to understand game log structure"""
        results, logger = self.evaluator.evaluate_agent_vs_agent_with_logger(
            agent1, agent2, game_class, 2  # Just 2 games for debugging
        )
        
        print("=== DEBUG: Game Logs Structure ===")
        game_logs = logger.game_logs_data
        print(f"Top level keys: {list(game_logs.keys())}")
        
        games = game_logs.get("games", {})
        print(f"Number of games: {len(games)}")
        
        for game_id, game_data in list(games.items())[:1]:  # Just first game
            print(f"\nGame {game_id} structure:")
            print(f"Keys: {list(game_data.keys())}")
            
            moves = game_data.get("moves", [])
            print(f"Number of moves: {len(moves)}")
            
            if moves:
                print(f"First move structure: {list(moves[0].keys())}")
                print(f"First move data: {moves[0]}")
            
            outcome = game_data.get("outcome")
            print(f"Outcome: {outcome}")
            print(f"Outcome type: {type(outcome)}")
            
            agent1_player = game_data.get("agent1_player_value")
            agent2_player = game_data.get("agent2_player_value")
            print(f"Agent1 player value: {agent1_player}")
            print(f"Agent2 player value: {agent2_player}")
        
    def calculate_balance_from_logs(self, game_logs: Dict) -> Dict[str, Any]:
        """Calculate balance metrics from game logs"""
        balance_score = 0.0
        first_player_wins = 0
        second_player_wins = 0
        total_valid_games = 0
        
        games = game_logs.get("games", {})
        
        for game_id, game_data in games.items():
            moves = game_data.get("moves", [])
            if not moves:
                continue
                
            # Get the outcome (winner information)
            outcome = game_data.get("outcome")
            if not outcome:
                continue
                
            # Parse winner from outcome string (e.g., "agent1_win", "agent2_win", "draw")
            winner_agent = None
            if "agent1" in outcome and ("win" in outcome or "wins" in outcome):
                winner_agent = "agent1"
            elif "agent2" in outcome and ("win" in outcome or "wins" in outcome):
                winner_agent = "agent2"
            elif "draw" in outcome or "Draw" in outcome or "tie" in outcome:
                continue  # Skip draws for balance calculation
            else:
                continue  # Skip unknown outcomes
                
            total_valid_games += 1
            
            # Determine who went first by looking at the first move
            first_move = moves[0]
            first_player_agent_tag = first_move.get("agent")
            
            # Get the agent-to-player mapping for this game
            agent1_player = game_data.get("agent1_player_value", 1)
            agent2_player = game_data.get("agent2_player_value", -1)
            
            # Determine which player went first
            if first_player_agent_tag == "agent1":
                first_player_value = agent1_player
            else:
                first_player_value = agent2_player
            
            # Determine which player value won
            if winner_agent == "agent1":
                winner_player_value = agent1_player
            else:
                winner_player_value = agent2_player
            
            # Count wins based on who went first
            if winner_player_value == first_player_value:
                first_player_wins += 1
            else:
                second_player_wins += 1
        
        if total_valid_games > 0:
            first_player_rate = first_player_wins / total_valid_games
            second_player_rate = second_player_wins / total_valid_games
            balance_score = first_player_rate - second_player_rate
        
        return {
            "balance_score": balance_score,
            "first_player_wins": first_player_wins,
            "second_player_wins": second_player_wins,
            "total_valid_games": total_valid_games,
            "first_player_rate": first_player_wins / total_valid_games if total_valid_games > 0 else 0,
            "second_player_rate": second_player_wins / total_valid_games if total_valid_games > 0 else 0
        }
    
    def calculate_deterministic_from_logs(self, results: Dict) -> Dict[str, Any]:
        """Calculate deterministic metrics from results"""
        total_games = results.get("total_games", 0)
        draws = results.get("draws", 0)
        
        non_draw_rate = (total_games - draws) / total_games if total_games > 0 else 0
        
        return {
            "non_draw_rate": non_draw_rate,
            "draws": draws,
            "total_games": total_games,
            "deterministic_score": non_draw_rate
        }
    
    def calculate_length_from_logs(self, game_logs: Dict) -> Dict[str, Any]:
        """Calculate length metrics from game logs"""
        games = game_logs.get("games", {})
        game_lengths = []
        
        for game_id, game_data in games.items():
            moves = game_data.get("moves", [])
            if moves:
                game_lengths.append(len(moves))
        
        if game_lengths:
            avg_length = statistics.mean(game_lengths)
            median_length = statistics.median(game_lengths)
            min_length = min(game_lengths)
            max_length = max(game_lengths)
            std_length = statistics.stdev(game_lengths) if len(game_lengths) > 1 else 0
        else:
            avg_length = median_length = min_length = max_length = std_length = 0
        
        return {
            "average_length": avg_length,
            "median_length": median_length,
            "min_length": min_length,
            "max_length": max_length,
            "std_length": std_length,
            "total_games_analyzed": len(game_lengths)
        }
    
    def calculate_variation_from_logs(self, game_logs: Dict) -> Dict[str, Any]:
        """Calculate variation metrics from game logs - unique states vs total states"""
        games = game_logs.get("games", {})
        unique_states = set()
        total_states = 0
        
        for game_id, game_data in games.items():
            moves = game_data.get("moves", [])
            if moves:
                for move_data in moves:
                    # Use board_before as state representation
                    state = move_data.get("board_before")
                    if state:
                        total_states += 1
                        # Create a normalized state representation for hashing
                        state_lines = state.split('\n')
                        # Extract just the board part (before "Current turn:")
                        board_part = []
                        for line in state_lines:
                            if line.startswith("Current turn:"):
                                break
                            board_part.append(line.strip())
                        
                        if board_part:
                            state_signature = tuple(board_part)
                            unique_states.add(state_signature)
        
        uniqueness_ratio = len(unique_states) / total_states if total_states > 0 else 0
        
        return {
            "uniqueness_ratio": uniqueness_ratio,
            "unique_states": len(unique_states),
            "total_states": total_states,
            "variation_score": uniqueness_ratio
        }
    
    def evaluate_agent_pair_metrics(self, game_class: type, agent1, agent2, agent_pair_name: str) -> Dict[str, Any]:
        """Evaluate balance, deterministic, and length for a specific agent pair"""
        cache_key = f"{game_class.__name__}_{agent1.name}_{agent2.name}"
        results, logger = self.get_or_cache_game_logs(game_class, agent1, agent2, cache_key)
        
        game_logs = logger.game_logs_data
        
        return {
            "agent_pair": agent_pair_name,
            "balance": self.calculate_balance_from_logs(game_logs),
            "deterministic": self.calculate_deterministic_from_logs(results),
            "length": self.calculate_length_from_logs(game_logs),
            "detailed_results": results
        }
    
    def evaluate_controllability(self, game_class: type, num_samples: int = 100) -> Dict[str, Any]:
        """
        Evaluate controllability: average legal moves per state
        
        Returns:
            Dict with controllability metrics
        """
        legal_move_counts = []
        
        # Sample random game states
        for _ in range(num_samples):
            game = game_class()
            
            # Play random moves until game ends or we sample enough states
            max_moves = 50  # Prevent infinite games
            moves_made = 0
            
            while not game.is_game_over() and moves_made < max_moves:
                legal_moves = game.get_legal_moves()
                if not legal_moves:
                    break
                    
                legal_move_counts.append(len(legal_moves))
                
                # Make a random move to continue
                import random
                move = random.choice(legal_moves)
                game.make_move(move)
                moves_made += 1
        
        if legal_move_counts:
            avg_legal_moves = statistics.mean(legal_move_counts)
            median_legal_moves = statistics.median(legal_move_counts)
            min_legal_moves = min(legal_move_counts)
            max_legal_moves = max(legal_move_counts)
            std_legal_moves = statistics.stdev(legal_move_counts) if len(legal_move_counts) > 1 else 0
        else:
            avg_legal_moves = median_legal_moves = min_legal_moves = max_legal_moves = std_legal_moves = 0
        
        return {
            "average_legal_moves": avg_legal_moves,
            "median_legal_moves": median_legal_moves,
            "min_legal_moves": min_legal_moves,
            "max_legal_moves": max_legal_moves,
            "std_legal_moves": std_legal_moves,
            "states_sampled": len(legal_move_counts),
            "controllability_score": avg_legal_moves
        }
    
    def evaluate_strategy_depth(self, game_class: type) -> Dict[str, Any]:
        """
        Evaluate strategy depth: random vs minimax depth 1-4, and depth battles
        
        Returns:
            Dict with strategy depth metrics
        """
        strategy_results = {}
        
        # Random vs Minimax depths 1-4
        random_agent = RandomAgent(name="Random")
        
        for depth in range(1, 5):
            minimax_agent = MinimaxAgent(name=f"Minimax-depth-{depth}", depth=depth)
            
            # Random vs Minimax
            self.logger.info(f"Evaluating Random vs Minimax depth {depth}")
            results, _ = self.evaluator.evaluate_agent_vs_agent_with_logger(
                random_agent, minimax_agent, game_class, self.num_games
            )
            
            minimax_wins = results.get("wins_agent2", 0)
            total_valid = results.get("total_games", 0) - results.get("forfeits_agent1", 0) - results.get("forfeits_agent2", 0)
            minimax_win_rate = minimax_wins / total_valid if total_valid > 0 else 0
            
            strategy_results[f"random_vs_minimax_depth_{depth}"] = {
                "minimax_win_rate": minimax_win_rate,
                "minimax_wins": minimax_wins,
                "total_valid_games": total_valid,
                "detailed_results": results
            }
        
        # Minimax depth battles: depth i vs depth i+1
        for depth1 in range(1, 4):
            depth2 = depth1 + 1
            
            agent1 = MinimaxAgent(name=f"Minimax-depth-{depth1}", depth=depth1)
            agent2 = MinimaxAgent(name=f"Minimax-depth-{depth2}", depth=depth2)
            
            self.logger.info(f"Evaluating Minimax depth {depth1} vs depth {depth2}")
            results, _ = self.evaluator.evaluate_agent_vs_agent_with_logger(
                agent1, agent2, game_class, self.num_games
            )
            
            higher_depth_wins = results.get("wins_agent2", 0)
            total_valid = results.get("total_games", 0) - results.get("forfeits_agent1", 0) - results.get("forfeits_agent2", 0)
            higher_depth_win_rate = higher_depth_wins / total_valid if total_valid > 0 else 0
            
            strategy_results[f"depth_{depth1}_vs_depth_{depth2}"] = {
                "higher_depth_win_rate": higher_depth_win_rate,
                "higher_depth_wins": higher_depth_wins,
                "lower_depth_wins": results.get("wins_agent1", 0),
                "total_valid_games": total_valid,
                "detailed_results": results
            }
        
        return strategy_results
    
    def evaluate_variation(self, game_class: type, temperature_values: List[float] = None) -> Dict[str, Any]:
        """
        Evaluate variation: minimax with temperature, count unique states ratio
        
        Returns:
            Dict with variation metrics
        """
        if temperature_values is None:
            temperature_values = [0.0, 0.5, 1.0, 2.0]
        
        variation_results = {}
        
        for temp in temperature_values:
            self.logger.info(f"Evaluating minimax with temperature {temp}")
            
            # Create minimax agents with temperature
            agent1 = MinimaxAgent(name=f"Minimax-temp-{temp}", depth=3, temperature=temp)
            agent2 = MinimaxAgent(name=f"Minimax-temp-{temp}-2", depth=3, temperature=temp)
            
            cache_key = f"{game_class.__name__}_temp_{temp}_variation"
            results, logger = self.get_or_cache_game_logs(game_class, agent1, agent2, cache_key)
            
            # Calculate variation metrics
            variation_metrics = self.calculate_variation_from_logs(logger.game_logs_data)
            variation_metrics["detailed_results"] = results
            
            variation_results[f"temperature_{temp}"] = variation_metrics
        
        return variation_results
    
    def evaluate_all_metrics(self, game_class: type) -> Dict[str, Any]:
        """
        Evaluate all metrics for a given game class efficiently by sampling once per agent pair
        
        Returns:
            Complete evaluation results
        """
        game_name = game_class.__name__
        self.logger.info(f"Starting comprehensive evaluation for {game_name}")
        
        all_results = {
            "game_name": game_name,
            "evaluation_timestamp": datetime.now().isoformat(),
            "num_games_per_metric": self.num_games
        }
        
        # Create agent pairs for evaluation
        agent_pairs = []
        
        # Random agents
        random_agent1 = RandomAgent(name="Random-1")
        random_agent2 = RandomAgent(name="Random-2")
        agent_pairs.append((random_agent1, random_agent2, "Random vs Random"))
        
        # Minimax agents at different depths
        for depth in range(1, 5):
            minimax_agent1 = MinimaxAgent(name=f"Minimax-depth-{depth}-1", depth=depth)
            minimax_agent2 = MinimaxAgent(name=f"Minimax-depth-{depth}-2", depth=depth)
            agent_pairs.append((minimax_agent1, minimax_agent2, f"Minimax depth {depth} vs depth {depth}"))
        
        # Evaluate balance, deterministic, and length for all agent pairs
        self.logger.info("Evaluating balance, deterministic, and length for all agent pairs...")
        agent_pair_results = []
        
        for agent1, agent2, pair_name in agent_pairs:
            self.logger.info(f"Evaluating {pair_name}")
            pair_results = self.evaluate_agent_pair_metrics(game_class, agent1, agent2, pair_name)
            agent_pair_results.append(pair_results)
        
        all_results["agent_pair_evaluations"] = agent_pair_results
        
        # Controllability evaluation (sampling game states)
        self.logger.info("Evaluating controllability...")
        all_results["controllability"] = self.evaluate_controllability(game_class)
        
        # Strategy depth evaluation
        self.logger.info("Evaluating strategy depth...")
        all_results["strategy_depth"] = self.evaluate_strategy_depth(game_class)
        
        # Variation evaluation
        self.logger.info("Evaluating variation...")
        all_results["variation"] = self.evaluate_variation(game_class)
        
        self.logger.info(f"Completed comprehensive evaluation for {game_name}")
        return all_results
    
    def print_summary(self, results: Dict[str, Any]):
        """Print a summary of evaluation results"""
        game_name = results.get("game_name", "Unknown")
        
        print(f"\n{'='*60}")
        print(f"EVALUATION SUMMARY FOR {game_name.upper()}")
        print(f"{'='*60}")
        
        # Agent pair evaluations (Balance, Deterministic, Length)
        agent_pairs = results.get("agent_pair_evaluations", [])
        if agent_pairs:
            print(f"\n1. AGENT PAIR EVALUATIONS:")
            for pair_data in agent_pairs:
                pair_name = pair_data.get("agent_pair", "Unknown")
                balance = pair_data.get("balance", {})
                deterministic = pair_data.get("deterministic", {})
                length = pair_data.get("length", {})
                
                print(f"\n   {pair_name}:")
                print(f"     Balance score: {balance.get('balance_score', 0):.3f}")
                print(f"     First player rate: {balance.get('first_player_rate', 0):.3f}")
                print(f"     Non-draw rate: {deterministic.get('non_draw_rate', 0):.3f}")
                print(f"     Average length: {length.get('average_length', 0):.1f} moves")
        
        # Controllability
        controllability = results.get("controllability", {})
        print(f"\n2. CONTROLLABILITY:")
        print(f"   Average legal moves: {controllability.get('average_legal_moves', 0):.1f}")
        print(f"   Range: {controllability.get('min_legal_moves', 0)}-{controllability.get('max_legal_moves', 0)}")
        print(f"   States sampled: {controllability.get('states_sampled', 0)}")
        
        # Strategy depth
        strategy = results.get("strategy_depth", {})
        print(f"\n3. STRATEGY DEPTH:")
        for depth in range(1, 5):
            key = f"random_vs_minimax_depth_{depth}"
            if key in strategy:
                win_rate = strategy[key].get("minimax_win_rate", 0)
                print(f"   Random vs Minimax-{depth}: {win_rate:.3f} minimax win rate")
        
        print(f"\n   Depth battles (higher depth win rate):")
        for depth1 in range(1, 4):
            depth2 = depth1 + 1
            key = f"depth_{depth1}_vs_depth_{depth2}"
            if key in strategy:
                win_rate = strategy[key].get("higher_depth_win_rate", 0)
                print(f"   Depth-{depth1} vs Depth-{depth2}: {win_rate:.3f}")
        
        # Variation
        variation = results.get("variation", {})
        print(f"\n4. VARIATION (unique states ratio):")
        for temp in [0.0, 0.5, 1.0, 2.0]:
            key = f"temperature_{temp}"
            if key in variation:
                uniqueness = variation[key].get("uniqueness_ratio", 0)
                unique_states = variation[key].get("unique_states", 0)
                total_states = variation[key].get("total_states", 0)
                print(f"   Temperature {temp}: {uniqueness:.3f} ({unique_states}/{total_states} states)")
    
    def save_results(self, results: Dict[str, Any], output_path: str = None):
        """Save evaluation results to JSON file"""
        if output_path is None:
            game_name = results.get("game_name", "unknown")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"debug/board_game_evaluation_{game_name}_{timestamp}.json"
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        self.logger.info(f"Results saved to: {output_path}")
        return output_path


def main():
    """Main evaluation function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Comprehensive BoardGame Evaluation")
    parser.add_argument("--game", type=str, default="all", 
                        help="Game to evaluate (or 'all' for all board games)")
    parser.add_argument("--num_games", type=int, default=30,
                        help="Number of games per evaluation metric")
    parser.add_argument("--output_dir", type=str, default="debug/evaluations",
                        help="Output directory for results")
    
    args = parser.parse_args()
    
    # Setup
    config = Config()
    evaluator = BoardGameBalanceEvaluator(config=config, num_games=args.num_games)
    
    # Determine games to evaluate
    if args.game == "all":
        games_to_eval = [game_class for game_class in Games.values() 
                        if issubclass(game_class, BoardGame)]
        print(f"Evaluating all BoardGame subclasses: {[g.__name__ for g in games_to_eval]}")
    else:
        try:
            game_class = GameByName(args.game)
            if not issubclass(game_class, BoardGame):
                print(f"Warning: {args.game} is not a BoardGame subclass")
            games_to_eval = [game_class]
        except ValueError as e:
            print(f"Error: {e}")
            return 1
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Evaluate each game
    all_results = []
    
    for game_class in games_to_eval:
        try:
            print(f"\n{'='*80}")
            print(f"EVALUATING {game_class.__name__.upper()}")
            print(f"{'='*80}")
            
            results = evaluator.evaluate_all_metrics(game_class)
            evaluator.print_summary(results)
            
            # Save individual results
            output_path = os.path.join(
                args.output_dir, 
                f"{game_class.__name__}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            evaluator.save_results(results, output_path)
            
            all_results.append(results)
            
        except Exception as e:
            print(f"Error evaluating {game_class.__name__}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save combined results
    if all_results:
        combined_results = {
            "evaluation_summary": {
                "timestamp": datetime.now().isoformat(),
                "num_games_per_metric": args.num_games,
                "total_games_evaluated": len(all_results)
            },
            "results": all_results
        }
        
        combined_path = os.path.join(
            args.output_dir,
            f"combined_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        
        with open(combined_path, 'w') as f:
            json.dump(combined_results, f, indent=2, default=str)
        
        print(f"\n{'='*80}")
        print(f"EVALUATION COMPLETE")
        print(f"{'='*80}")
        print(f"Individual results saved in: {args.output_dir}")
        print(f"Combined results saved to: {combined_path}")
        print(f"Total games evaluated: {len(all_results)}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
