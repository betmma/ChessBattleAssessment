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
    

    

    
    def generate_all_game_logs(self, game_class: type) -> Dict[str, Tuple[Dict, Any]]:
        """Generate all game logs needed for evaluation"""
        game_name = game_class.__name__
        self.logger.info(f"Starting game log generation for {game_name}")
        
        all_logs = {}
        
        # Random vs Random
        self.logger.info("Generating Random vs Random logs...")
        agent1 = RandomAgent(name="Random-1")
        agent2 = RandomAgent(name="Random-2")
        cache_key = f"{game_name}_Random_vs_Random"
        all_logs["Random vs Random"] = self.get_or_cache_game_logs(game_class, agent1, agent2, cache_key)
        
        # Minimax vs Minimax (same depth)
        for depth in range(1, self.depth + 1):
            self.logger.info(f"Generating Minimax depth {depth} vs depth {depth} logs...")
            agent1 = MinimaxAgent(name=f"Minimax-depth-{depth}-1", depth=depth)
            agent2 = MinimaxAgent(name=f"Minimax-depth-{depth}-2", depth=depth)
            cache_key = f"{game_name}_Minimax-depth-{depth}_vs_Minimax-depth-{depth}"
            all_logs[f"Minimax depth {depth} vs depth {depth}"] = self.get_or_cache_game_logs(game_class, agent1, agent2, cache_key)
        
        # Random vs Minimax (for strategy depth)
        random_agent = RandomAgent(name="Random")
        for depth in range(1, self.depth + 1):
            self.logger.info(f"Generating Random vs Minimax depth {depth} logs...")
            minimax_agent = MinimaxAgent(name=f"Minimax-depth-{depth}", depth=depth)
            cache_key = f"{game_name}_Random_vs_Minimax-depth-{depth}"
            all_logs[f"Random vs Minimax depth {depth}"] = self.get_or_cache_game_logs(game_class, random_agent, minimax_agent, cache_key)
        
        # Minimax depth battles (depth i vs depth i+1)
        for depth1 in range(1, self.depth):
            depth2 = depth1 + 1
            self.logger.info(f"Generating Minimax depth {depth1} vs depth {depth2} logs...")
            agent1 = MinimaxAgent(name=f"Minimax-depth-{depth1}", depth=depth1)
            agent2 = MinimaxAgent(name=f"Minimax-depth-{depth2}", depth=depth2)
            cache_key = f"{game_name}_Minimax-depth-{depth1}_vs_Minimax-depth-{depth2}"
            all_logs[f"Minimax depth {depth1} vs depth {depth2}"] = self.get_or_cache_game_logs(game_class, agent1, agent2, cache_key)
        
        # Minimax with temperature (for variation)
        temperature_values = [0.0, 0.5, 1.0, 2.0]
        for temp in temperature_values:
            self.logger.info(f"Generating Minimax temperature {temp} vs temperature {temp} logs...")
            agent1 = MinimaxAgent(name=f"Minimax-temp-{temp}-1", depth=2, temperature=temp)
            agent2 = MinimaxAgent(name=f"Minimax-temp-{temp}-2", depth=2, temperature=temp)
            cache_key = f"{game_name}_Minimax-temp-{temp}_vs_Minimax-temp-{temp}"
            all_logs[f"Minimax temperature {temp} vs temperature {temp}"] = self.get_or_cache_game_logs(game_class, agent1, agent2, cache_key)
        
        self.logger.info(f"Completed game log generation for {game_name}")
        return all_logs
    
    def calculate_all_metrics_from_logs(self, game_logs_dict: Dict[str, Tuple[Dict, Any]]) -> Dict[str, Dict[str, Any]]:
        """Calculate all metrics from generated game logs"""
        self.logger.info("Calculating metrics from game logs...")
        
        agent_pair_metrics = {}
        
        for pair_name, (results, logger) in game_logs_dict.items():
            self.logger.info(f"Calculating metrics for {pair_name}")
            game_logs = logger.game_logs_data
            
            # Calculate all metrics for this agent pair
            metrics = {
                "balance": self.calculate_balance_from_logs(game_logs),
                "deterministic": self.calculate_deterministic_from_logs(results),
                "length": self.calculate_length_from_logs(game_logs),
                "variation": self.calculate_variation_from_logs(game_logs),
                "detailed_results": results
            }
            
            agent_pair_metrics[pair_name] = metrics
        
        self.logger.info("Completed metric calculations")
        return agent_pair_metrics
    
    def evaluate_all_metrics(self, game_class: type) -> Dict[str, Any]:
        game_name = game_class.__name__
        self.logger.info(f"Starting comprehensive evaluation for {game_name}")
        
        # Step 1: Generate all game logs
        all_game_logs = self.generate_all_game_logs(game_class)
        
        # Step 2: Calculate metrics from logs
        agent_pair_metrics = self.calculate_all_metrics_from_logs(all_game_logs)
        
        # Step 3: Calculate controllability (doesn't need game logs)
        self.logger.info("Evaluating controllability...")
        controllability = self.evaluate_controllability(game_class)
        
        all_results = {
            "game_name": game_name,
            "evaluation_timestamp": datetime.now().isoformat(),
            "num_games_per_metric": self.num_games,
            "agent_pair_metrics": agent_pair_metrics,
            "controllability": controllability
        }
        
        self.logger.info(f"Completed comprehensive evaluation for {game_name}")
        return all_results
    
    def format_summary(self, results: Dict[str, Any]) -> str:
        game_name = results.get("game_name", "Unknown")
        summary_lines = [f"\n{'='*60}\nEVALUATION SUMMARY FOR {game_name.upper()}\n{'='*60}"]
        
        # Agent pair metrics
        agent_pair_metrics = results.get("agent_pair_metrics", {})
        if agent_pair_metrics:
            summary_lines.append(f"\nAGENT PAIR METRICS:")
            
            for pair_name, metrics in agent_pair_metrics.items():
                balance = metrics.get("balance", {})
                deterministic = metrics.get("deterministic", {})
                length = metrics.get("length", {})
                variation = metrics.get("variation", {})
                detailed_results = metrics.get("detailed_results", {})
                
                # Calculate win rates from detailed results
                total_games = detailed_results.get("total_games", 0)
                agent1_wins = detailed_results.get("wins_agent1", 0)
                agent2_wins = detailed_results.get("wins_agent2", 0)
                draws = detailed_results.get("draws", 0)
                
                agent1_win_rate = agent1_wins / total_games if total_games > 0 else 0
                agent2_win_rate = agent2_wins / total_games if total_games > 0 else 0
                draw_rate = draws / total_games if total_games > 0 else 0
                
                summary_lines.extend([
                    f"\n{pair_name}:",
                    f"  Agent1 win rate: {agent1_win_rate:.3f} ({agent1_wins}/{total_games})",
                    f"  Agent2 win rate: {agent2_win_rate:.3f} ({agent2_wins}/{total_games})",
                    f"  Draw rate: {draw_rate:.3f} ({draws}/{total_games})",
                    f"  Balance score (0 is best): {balance.get('balance_score', 0):.3f}",
                    f"  First player win rate: {balance.get('first_player_rate', 0):.3f}",
                    f"  Second player win rate: {balance.get('second_player_rate', 0):.3f}",
                    f"  Deterministic score: {deterministic.get('deterministic_score', 0):.3f}",
                    f"  Average length: {length.get('average_length', 0):.1f} moves",
                    f"  Variation score: {variation.get('variation_score', 0):.3f}"
                ])
        
        # Controllability
        controllability = results.get("controllability", {})
        summary_lines.extend([
            f"\nCONTROLLABILITY:",
            f"  Average legal moves: {controllability.get('average_legal_moves', 0):.1f}",
            f"  Range: {controllability.get('min_legal_moves', 0)}-{controllability.get('max_legal_moves', 0)}",
            f"  States sampled: {controllability.get('states_sampled', 0)}",
            f"  Controllability score: {controllability.get('controllability_score', 0):.1f}"
        ])
        
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
    config.LOG_ACTION_REWARDS=False
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
