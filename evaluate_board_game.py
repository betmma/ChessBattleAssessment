"""BoardGame evaluation program for balance, deterministic, length, controllability, strategy depth, and variation"""
import os, sys, statistics, json, argparse
from datetime import datetime
from typing import Dict, List, Tuple, Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import Config, setup_logging
from games import Games, GameByName
from agents import RandomAgent, MinimaxAgent
from evaluation.evaluator import Evaluator
from games.board_game import BoardGame

class BoardGameBalanceEvaluator:
    def __init__(self, config: Config = None, num_games: int = 50, depth: int = 2):
        self.config = config or Config()
        self.num_games = num_games
        self.depth = depth
        self.evaluator = Evaluator(config=self.config, retry_limit=3)
        self.logger = setup_logging(self.config)
        self.game_logs_cache = {}
    
    def get_or_cache_game_logs(self, game_class: type, agent1, agent2, cache_key: str) -> Tuple[Dict, Any]:
        if cache_key in self.game_logs_cache:
            return self.game_logs_cache[cache_key]
        results, logger = self.evaluator.evaluate_agent_vs_agent_with_logger(agent1, agent2, game_class, self.num_games)
        self.game_logs_cache[cache_key] = (results, logger)
        return results, logger
        
    def calculate_balance_from_logs(self, game_logs: Dict) -> Dict[str, Any]:
        first_player_wins = second_player_wins = total_valid_games = 0
        
        for game_data in game_logs.get("games", {}).values():
            moves = game_data.get("moves", [])
            outcome = game_data.get("outcome")
            if not moves or not outcome or "draw" in outcome.lower():
                continue
                
            winner_agent = "agent1" if "agent1" in outcome and "win" in outcome else ("agent2" if "agent2" in outcome and "win" in outcome else None)
            if not winner_agent:
                continue
                
            total_valid_games += 1
            first_player_agent_tag = moves[0].get("agent")
            agent1_player = game_data.get("agent1_player_value", 1)
            agent2_player = game_data.get("agent2_player_value", -1)
            
            first_player_value = agent1_player if first_player_agent_tag == "agent1" else agent2_player
            winner_player_value = agent1_player if winner_agent == "agent1" else agent2_player
            
            if winner_player_value == first_player_value:
                first_player_wins += 1
            else:
                second_player_wins += 1
        
        balance_score = (first_player_wins - second_player_wins) / total_valid_games if total_valid_games > 0 else 0
        return {
            "balance_score": balance_score,
            "first_player_wins": first_player_wins,
            "second_player_wins": second_player_wins,
            "total_valid_games": total_valid_games,
            "first_player_rate": first_player_wins / total_valid_games if total_valid_games > 0 else 0,
            "second_player_rate": second_player_wins / total_valid_games if total_valid_games > 0 else 0
        }
    
    def calculate_deterministic_from_logs(self, results: Dict) -> Dict[str, Any]:
        total_games = results.get("total_games", 0)
        draws = results.get("draws", 0)
        non_draw_rate = (total_games - draws) / total_games if total_games > 0 else 0
        return {"non_draw_rate": non_draw_rate, "draws": draws, "total_games": total_games, "deterministic_score": non_draw_rate}
    
    def calculate_length_from_logs(self, game_logs: Dict) -> Dict[str, Any]:
        game_lengths = [len(game_data.get("moves", [])) for game_data in game_logs.get("games", {}).values() if game_data.get("moves")]
        if game_lengths:
            return {
                "average_length": statistics.mean(game_lengths),
                "median_length": statistics.median(game_lengths),
                "min_length": min(game_lengths),
                "max_length": max(game_lengths),
                "std_length": statistics.stdev(game_lengths) if len(game_lengths) > 1 else 0,
                "total_games_analyzed": len(game_lengths)
            }
        return {"average_length": 0, "median_length": 0, "min_length": 0, "max_length": 0, "std_length": 0, "total_games_analyzed": 0}
    
    def calculate_variation_from_logs(self, game_logs: Dict) -> Dict[str, Any]:
        unique_states, total_states = set(), 0
        for game_data in game_logs.get("games", {}).values():
            for move in game_data.get("moves", []):
                state = move.get("board_before", "")
                if state:
                    total_states += 1
                    state_lines = state.split('\n')
                    board_part = []
                    for line in state_lines:
                        if line.startswith("Current turn:"):
                            break
                        board_part.append(line.strip())
                    if board_part:
                        unique_states.add(tuple(board_part))
        uniqueness_ratio = len(unique_states) / total_states if total_states > 0 else 0
        return {"uniqueness_ratio": uniqueness_ratio, "unique_states": len(unique_states), "total_states": total_states, "variation_score": uniqueness_ratio}
    
    def evaluate_agent_pair_metrics(self, game_class: type, agent1, agent2, agent_pair_name: str) -> Dict[str, Any]:
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
        legal_move_counts = []
        for _ in range(num_samples):
            game = game_class()
            max_moves = 50
            moves_made = 0
            while not game.is_game_over() and moves_made < max_moves:
                legal_moves = game.get_legal_moves()
                if not legal_moves:
                    break
                legal_move_counts.append(len(legal_moves))
                import random
                game.make_move(random.choice(legal_moves))
                moves_made += 1
        
        if legal_move_counts:
            return {
                "average_legal_moves": statistics.mean(legal_move_counts),
                "median_legal_moves": statistics.median(legal_move_counts),
                "min_legal_moves": min(legal_move_counts),
                "max_legal_moves": max(legal_move_counts),
                "std_legal_moves": statistics.stdev(legal_move_counts) if len(legal_move_counts) > 1 else 0,
                "states_sampled": len(legal_move_counts),
                "controllability_score": statistics.mean(legal_move_counts)
            }
        return {"average_legal_moves": 0, "median_legal_moves": 0, "min_legal_moves": 0, "max_legal_moves": 0, "std_legal_moves": 0, "states_sampled": 0, "controllability_score": 0}
    
    def evaluate_strategy_depth(self, game_class: type) -> Dict[str, Any]:
        strategy_results = {}
        random_agent = RandomAgent(name="Random")
        
        for depth in range(1, self.depth + 1):
            minimax_agent = MinimaxAgent(name=f"Minimax-depth-{depth}", depth=depth)
            results, _ = self.evaluator.evaluate_agent_vs_agent_with_logger(random_agent, minimax_agent, game_class, self.num_games)
            minimax_wins = results.get("wins_agent2", 0)
            total_valid = results.get("total_games", 0) - results.get("forfeits_agent1", 0) - results.get("forfeits_agent2", 0)
            strategy_results[f"random_vs_minimax_depth_{depth}"] = {
                "minimax_win_rate": minimax_wins / total_valid if total_valid > 0 else 0,
                "minimax_wins": minimax_wins,
                "total_valid_games": total_valid
            }
        
        for depth1 in range(1, self.depth):
            depth2 = depth1 + 1
            agent1 = MinimaxAgent(name=f"Minimax-depth-{depth1}", depth=depth1)
            agent2 = MinimaxAgent(name=f"Minimax-depth-{depth2}", depth=depth2)
            results, _ = self.evaluator.evaluate_agent_vs_agent_with_logger(agent1, agent2, game_class, self.num_games)
            higher_depth_wins = results.get("wins_agent2", 0)
            lower_depth_wins = results.get("wins_agent1", 0)
            draws = results.get("draws", 0)
            total_valid = results.get("total_games", 0) - results.get("forfeits_agent1", 0) - results.get("forfeits_agent2", 0)
            
            strategy_results[f"depth_{depth1}_vs_depth_{depth2}"] = {
                "higher_depth_win_rate": higher_depth_wins / total_valid if total_valid > 0 else 0,
                "lower_depth_win_rate": lower_depth_wins / total_valid if total_valid > 0 else 0,
                "draw_rate": draws / total_valid if total_valid > 0 else 0,
                "higher_depth_wins": higher_depth_wins,
                "lower_depth_wins": lower_depth_wins,
                "draws": draws,
                "total_valid_games": total_valid
            }
        return strategy_results
    
    def evaluate_variation(self, game_class: type, temperature_values: List[float] = None) -> Dict[str, Any]:
        if temperature_values is None:
            temperature_values = [0.0, 0.5, 1.0, 2.0]
        
        variation_results = {}
        for temp in temperature_values:
            agent1 = MinimaxAgent(name=f"Minimax-temp-{temp}", depth=2, temperature=temp)
            agent2 = MinimaxAgent(name=f"Minimax-temp-{temp}-2", depth=2, temperature=temp)
            cache_key = f"{game_class.__name__}_temp_{temp}_variation"
            results, logger = self.get_or_cache_game_logs(game_class, agent1, agent2, cache_key)
            variation_metrics = self.calculate_variation_from_logs(logger.game_logs_data)
            variation_metrics["detailed_results"] = results
            variation_results[f"temperature_{temp}"] = variation_metrics
        return variation_results
    
    def evaluate_all_metrics(self, game_class: type) -> Dict[str, Any]:
        game_name = game_class.__name__
        self.logger.info(f"Starting comprehensive evaluation for {game_name}")
        
        all_results = {
            "game_name": game_name,
            "evaluation_timestamp": datetime.now().isoformat(),
            "num_games_per_metric": self.num_games
        }
        
        # Agent pairs for evaluation
        agent_pairs = [(RandomAgent(name="Random-1"), RandomAgent(name="Random-2"), "Random vs Random")]
        for depth in range(1, self.depth + 1):
            agent_pairs.append((MinimaxAgent(name=f"Minimax-depth-{depth}-1", depth=depth), 
                              MinimaxAgent(name=f"Minimax-depth-{depth}-2", depth=depth), 
                              f"Minimax depth {depth} vs depth {depth}"))
        
        # Evaluate balance, deterministic, and length
        agent_pair_results = []
        for agent1, agent2, pair_name in agent_pairs:
            self.logger.info(f"Evaluating {pair_name}")
            pair_results = self.evaluate_agent_pair_metrics(game_class, agent1, agent2, pair_name)
            agent_pair_results.append(pair_results)
        
        all_results["agent_pair_evaluations"] = agent_pair_results
        
        # Other evaluations
        self.logger.info("Evaluating controllability...")
        all_results["controllability"] = self.evaluate_controllability(game_class)
        
        self.logger.info("Evaluating strategy depth...")
        all_results["strategy_depth"] = self.evaluate_strategy_depth(game_class)
        
        self.logger.info("Evaluating variation...")
        all_results["variation"] = self.evaluate_variation(game_class)
        
        self.logger.info(f"Completed comprehensive evaluation for {game_name}")
        return all_results
    
    def format_summary(self, results: Dict[str, Any]) -> str:
        game_name = results.get("game_name", "Unknown")
        summary_lines = [f"\n{'='*60}\nEVALUATION SUMMARY FOR {game_name.upper()}\n{'='*60}"]
        
        # Agent pair evaluations
        agent_pairs = results.get("agent_pair_evaluations", [])
        if agent_pairs:
            summary_lines.append(f"\n1. AGENT PAIR EVALUATIONS:")
            for pair_data in agent_pairs:
                pair_name = pair_data.get("agent_pair", "Unknown")
                balance = pair_data.get("balance", {})
                deterministic = pair_data.get("deterministic", {})
                length = pair_data.get("length", {})
                
                summary_lines.extend([
                    f"\n   {pair_name}:",
                    f"     Balance score (0 is best): {balance.get('balance_score', 0):.3f}",
                    f"     First player win rate: {balance.get('first_player_rate', 0):.3f}",
                    f"     Non-draw rate: {deterministic.get('non_draw_rate', 0):.3f}",
                    f"     Average length: {length.get('average_length', 0):.1f} moves"
                ])
        
        # Controllability
        controllability = results.get("controllability", {})
        summary_lines.extend([
            f"\n2. CONTROLLABILITY:",
            f"   Average legal moves: {controllability.get('average_legal_moves', 0):.1f}",
            f"   Range: {controllability.get('min_legal_moves', 0)}-{controllability.get('max_legal_moves', 0)}",
            f"   States sampled: {controllability.get('states_sampled', 0)}"
        ])
        
        # Strategy depth
        strategy = results.get("strategy_depth", {})
        summary_lines.append(f"\n3. STRATEGY DEPTH:")
        for depth in range(1, self.depth + 1):
            key = f"random_vs_minimax_depth_{depth}"
            if key in strategy:
                win_rate = strategy[key].get("minimax_win_rate", 0)
                summary_lines.append(f"   Random vs Minimax-{depth}: {win_rate:.3f} minimax win rate")
        
        summary_lines.append(f"\n   Depth battles (win/draw/lose rates for higher depth):")
        for depth1 in range(1, self.depth):
            depth2 = depth1 + 1
            key = f"depth_{depth1}_vs_depth_{depth2}"
            if key in strategy:
                win_rate = strategy[key].get("higher_depth_win_rate", 0)
                draw_rate = strategy[key].get("draw_rate", 0)
                lose_rate = strategy[key].get("lower_depth_win_rate", 0)
                wins = strategy[key].get("higher_depth_wins", 0)
                draws = strategy[key].get("draws", 0)
                losses = strategy[key].get("lower_depth_wins", 0)
                total = strategy[key].get("total_valid_games", 0)
                summary_lines.append(f"   Depth-{depth1} vs Depth-{depth2}: Win {win_rate:.3f} ({wins}/{total}), Draw {draw_rate:.3f} ({draws}/{total}), Lose {lose_rate:.3f} ({losses}/{total})")
        
        # Variation
        variation = results.get("variation", {})
        summary_lines.append(f"\n4. VARIATION (unique states ratio):")
        for temp in [0.0, 0.5, 1.0, 2.0]:
            key = f"temperature_{temp}"
            if key in variation:
                uniqueness = variation[key].get("uniqueness_ratio", 0)
                unique_states = variation[key].get("unique_states", 0)
                total_states = variation[key].get("total_states", 0)
                summary_lines.append(f"   Temperature {temp}: {uniqueness:.3f} ({unique_states}/{total_states} states)")
        
        return '\n'.join(summary_lines)
    
    def save_results(self, results: Dict[str, Any], output_path: str = None):
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"debug/evaluations/{results['game_name']}_evaluation_{timestamp}.json"
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        self.logger.info(f"Results saved to: {output_path}")
        return output_path

def main():
    parser = argparse.ArgumentParser(description="Comprehensive BoardGame Evaluation")
    parser.add_argument("--game", type=str, default="all", help="Game to evaluate (or 'all' for all board games)")
    parser.add_argument("--num_games", type=int, default=30, help="Number of games per evaluation metric")
    parser.add_argument("--output_dir", type=str, default="debug/evaluations", help="Output directory for results")
    parser.add_argument("--depth", type=int, default=2, help="Minimax depth for evaluation (default: 2)")
    
    args = parser.parse_args()
    
    config = Config()
    evaluator = BoardGameBalanceEvaluator(config=config, num_games=args.num_games, depth=args.depth)
    
    messages = []  # Collect all output messages
    
    if args.game == "all":
        games_to_eval = [game_class for game_class in Games.values() if issubclass(game_class, BoardGame)]
        messages.append(f"Evaluating all BoardGame subclasses: {[g.__name__ for g in games_to_eval]}")
    else:
        try:
            game_class = GameByName(args.game)
            if not issubclass(game_class, BoardGame):
                messages.append(f"Error: {args.game} is not a BoardGame subclass")
                print('\n'.join(messages))
                return 1
            games_to_eval = [game_class]
        except ValueError as e:
            messages.append(f"Error: {e}")
            print('\n'.join(messages))
            return 1
    
    os.makedirs(args.output_dir, exist_ok=True)
    all_results = []
    all_summaries = []
    
    for game_class in games_to_eval:
        try:
            messages.append(f"\n{'='*80}\nEvaluating {game_class.__name__}...\n{'='*80}")
            results = evaluator.evaluate_all_metrics(game_class)
            summary = evaluator.format_summary(results)
            all_summaries.append(summary)
            output_path = evaluator.save_results(results, os.path.join(args.output_dir, f"{game_class.__name__}_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"))
            all_results.append(results)
        except Exception as e:
            messages.append(f"Error evaluating {game_class.__name__}: {e}")
            import traceback
            messages.append(traceback.format_exc())
    
    if all_results:
        combined_results = {
            "evaluation_summary": {
                "timestamp": datetime.now().isoformat(),
                "num_games_per_metric": args.num_games,
                "total_games_evaluated": len(all_results)
            },
            "results": all_results,
            "formatted_summaries": all_summaries
        }
        
        combined_path = os.path.join(args.output_dir, f"combined_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(combined_path, 'w') as f:
            json.dump(combined_results, f, indent=2, default=str)
        
        # Save summary to separate file
        summary_path = os.path.join(args.output_dir, f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        with open(summary_path, 'w') as f:
            f.write('\n'.join(all_summaries))
        
        messages.extend([
            f"\n{'='*80}\nEVALUATION COMPLETE\n{'='*80}",
            f"Individual results saved in: {args.output_dir}",
            f"Combined results saved to: {combined_path}",
            f"Summary saved to: {summary_path}",
            f"Total games evaluated: {len(all_results)}"
        ])
    
    # Print all collected messages once
    print('\n'.join(messages))
    if all_summaries:
        print('\n'.join(all_summaries))
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
