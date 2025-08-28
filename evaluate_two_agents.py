import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import logging
import argparse
import sys
import math
from types import SimpleNamespace

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from config import Config, setup_logging
from games import GameByName, Games
from agents import RandomAgent, MinimaxAgent, APIAgent, VLLMAgent
from evaluation.evaluator import Evaluator, ConsolidatedLogger
from utils import create_agent

try:
    from openai import OpenAI
    import httpx
except ImportError:
    print("WARNING: OpenAI Python package not installed. Install with 'pip install openai httpx'")
    OpenAI = None
    httpx = None

def parse_args():
    parser = argparse.ArgumentParser(description="Two Agent Battle Evaluation for Multiple Games")
    parser.add_argument("--num_games", type=int, default=50, help="Number of games to evaluate per game type")
    parser.add_argument("--game", type=str, nargs='+', default="all", 
                        help="Game to evaluate (choose from available games or 'all' for all games)")
    
    # Agent 1 configuration
    parser.add_argument("--agent1", type=str, required=True, 
                        choices=["api", "random", "minimax", "vllm", "mcts"],
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
    parser.add_argument("--agent1_temperature", type=float, default=0.0,
                        help="Temperature of agent1 (only used if agent1 is 'minimax')")
    parser.add_argument("--agent1_max_depth", type=int, default=4,
                        help="Max depth for minimax agent (only used if agent1 is 'minimax')")
    
    # Agent 2 configuration
    parser.add_argument("--agent2", type=str, required=True, 
                        choices=["api", "random", "minimax", "vllm", "mcts"],
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
    parser.add_argument("--agent2_temperature", type=float, default=0.0,
                        help="Temperature of agent2 (only used if agent2 is 'minimax')")
    parser.add_argument("--agent2_max_depth", type=int, default=4,
                        help="Max depth for minimax agent (only used if agent2 is 'minimax')")
    
    parser.add_argument("--output_dir", type=str, help="Output directory for results")
    
    return parser.parse_args()


def calculate_wilson_ci(successes, trials, confidence=0.95):
    """
    Calculate Wilson score confidence interval for proportion.
    More accurate than normal approximation, especially for small samples.
    
    Args:
        successes: Number of successes
        trials: Total number of trials
        confidence: Confidence level (default 0.95 for 95% CI)
    
    Returns:
        tuple: (lower_bound, upper_bound) as percentages
    """
    if trials == 0:
        return (0.0, 0.0)
    
    p = successes / trials
    z = 1.96  # 95% confidence level
    
    denominator = 1 + z**2 / trials
    center = (p + z**2 / (2 * trials)) / denominator
    margin = z * math.sqrt((p * (1 - p) + z**2 / (4 * trials)) / trials) / denominator
    
    lower = max(0, center - margin) * 100
    upper = min(1, center + margin) * 100
    
    return (lower, upper)


def main():
    args = parse_args()
    
    # Initialize config
    config = Config()
    
    # Override config with command line arguments
    if args.num_games:
        config.NUM_EVAL_GAMES = args.num_games
    if args.output_dir:
        config.OUTPUT_DIR_BASE = args.output_dir
    
    # Setup logging with a consolidated log file
    logger = setup_logging(config)
    
    # Determine which games to run
    games_to_run = []
    if '/' in args.game[0]: # folder
        from utils.load_games_in_folder import load_games_in_folder
        games_to_run=load_games_in_folder(args.game[0])
    elif 'all' in args.game:
        games_to_run = list(Games.keys())
        logger.info(f"Running evaluation for ALL games: {games_to_run}")
    else:
        games_to_run = args.game
        logger.info(f"Running evaluation for game: {args.game}")
    
    logger.info(f"Starting two-agent battle evaluation: {args.agent1} vs {args.agent2}")
    
    # Create agent 1
    try:
        agent1 = create_agent(
            args.agent1, 
            model=args.agent1_model,
            api_base_url=args.agent1_api_base_url,
            api_key=args.agent1_api_key,
            model_path=args.agent1_model_path,
            temperature=args.agent1_temperature,
            max_depth=args.agent1_max_depth
        )
        logger.info(f"Successfully initialized Agent1: {agent1.name}")
    except Exception as e:
        logger.critical(f"Failed to initialize Agent1 ({args.agent1}): {e}")
        raise
        return 1
    
    # Create agent 2
    try:
        if args.agent2==args.agent1=='vllm' and args.agent2_model_path==args.agent1_model_path:
            agent2 = agent1
            logger.info(f"Agent2 is the same as Agent1: {agent2.name}")
        else:
            agent2 = create_agent(
                args.agent2, 
                model=args.agent2_model,
                api_base_url=args.agent2_api_base_url,
                api_key=args.agent2_api_key,
                model_path=args.agent2_model_path,
                temperature=args.agent2_temperature,
                max_depth=args.agent2_max_depth
            )
            logger.info(f"Successfully initialized Agent2: {agent2.name}")
    except Exception as e:
        logger.critical(f"Failed to initialize Agent2 ({args.agent2}): {e}")
        return 1
    
    # Initialize evaluator
    evaluator = Evaluator(config)
    
    # Initialize consolidated logger for multiple game types
    consolidated_logger = ConsolidatedLogger(agent1.name, agent2.name, config)
    
    # Store all results for final summary
    all_results = {}
    total_games_played = 0
    
    # Run one multi-game evaluation covering all selected game types
    try:
        game_classes = [GameByName(name)if type(name)==str else name for name in games_to_run]
        if type(games_to_run[0])!=str:
            games_to_run=[cls.__name__.removesuffix('Game') for cls in game_classes]
        overall_results, eval_logger = evaluator.evaluate_agent_vs_agent_with_logger(
            agent1, agent2, game_classes, config.NUM_EVAL_GAMES
        )

        # Aggregate per-game-type summaries from the unified logger
        per_game_acc = {}
        games_logs = eval_logger.game_logs_data.get("games", {})
        for gid, ginfo in games_logs.items():
            gname = ginfo.get("game_name") or eval_logger.game_class_name
            outcome = ginfo.get("outcome")
            acc = per_game_acc.setdefault(gname, {
                "wins_agent1": 0, "wins_agent2": 0, "draws": 0,
                "forfeits_agent1": 0, "forfeits_agent2": 0, "total_games": 0,
                "agent1_name": agent1.name, "agent2_name": agent2.name, "game_name": gname
            })
            acc["total_games"] += 1
            if outcome == "agent1_win":
                acc["wins_agent1"] += 1
            elif outcome == "agent2_win":
                acc["wins_agent2"] += 1
            elif outcome == "draw":
                acc["draws"] += 1
            elif outcome == "agent1_forfeit":
                acc["forfeits_agent1"] += 1
            elif outcome == "agent2_forfeit":
                acc["forfeits_agent2"] += 1
            else:
                # unknown or None, count towards total but not specific bucket
                pass
        
        # Compute rates and push into consolidated logger with filtered logs per game
        for gname, res in per_game_acc.items():
            total = res["total_games"]
            valid_completed = total - res["forfeits_agent1"] - res["forfeits_agent2"]
            res["win_rate_agent1"] = (res["wins_agent1"] / valid_completed) if valid_completed > 0 else 0
            res["win_rate_agent2"] = (res["wins_agent2"] / valid_completed) if valid_completed > 0 else 0
            res["draw_rate"] = (res["draws"] / valid_completed) if valid_completed > 0 else 0

            # Prepare filtered detailed logs for this game
            filtered_games = {gid: g for gid, g in games_logs.items() if (g.get("game_name") or eval_logger.game_class_name) == gname}
            filtered_log_data = {
                **eval_logger.game_logs_data,
                "game": gname,
                "games": filtered_games
            }
            filtered_logger = SimpleNamespace(game_logs_data=filtered_log_data)

            # Add to consolidated logger and local tracking
            consolidated_logger.add_game_evaluation_data(gname, filtered_logger, res)
            all_results[gname] = res
            total_games_played += total

            # Log results for this game with confidence intervals
            agent1_wins = res.get('wins_agent1', 0)
            agent2_wins = res.get('wins_agent2', 0)
            draws = res.get('draws', 0)
            valid_games_this_round = agent1_wins + agent2_wins + draws
            if valid_games_this_round > 0:
                agent1_rate = (agent1_wins / valid_games_this_round) * 100
                agent2_rate = (agent2_wins / valid_games_this_round) * 100
                draw_rate = (draws / valid_games_this_round) * 100
                agent1_ci = calculate_wilson_ci(agent1_wins, valid_games_this_round)
                agent2_ci = calculate_wilson_ci(agent2_wins, valid_games_this_round)
                draw_ci = calculate_wilson_ci(draws, valid_games_this_round)
                logger.info(f"Results for {gname}:")
                logger.info(f"  Agent1 wins: {agent1_wins}/{valid_games_this_round} ({agent1_rate:.1f}%, 95% CI: {agent1_ci[0]:.1f}%-{agent1_ci[1]:.1f}%)")
                logger.info(f"  Agent2 wins: {agent2_wins}/{valid_games_this_round} ({agent2_rate:.1f}%, 95% CI: {agent2_ci[0]:.1f}%-{agent2_ci[1]:.1f}%)")
                logger.info(f"  Draws: {draws}/{valid_games_this_round} ({draw_rate:.1f}%, 95% CI: {draw_ci[0]:.1f}%-{draw_ci[1]:.1f}%)")
            else:
                logger.info(f"Results for {gname}: {res}")

    except Exception as e:
        logger.error(f"Failed to run multi-game evaluation: {e}")
        for game_name in games_to_run:
            all_results[game_name] = {"error": str(e)}
    
    # Save consolidated logs to file
    if games_to_run and len([r for r in all_results.values() if isinstance(r, dict) and "error" not in r]) > 0:
        consolidated_log_path = consolidated_logger.save_consolidated_logs_to_file()
        consolidated_summary_path = consolidated_logger.save_consolidated_summary_report()
        logger.info(f"Consolidated logs saved to: {consolidated_log_path}")
        logger.info(f"Consolidated summary saved to: {consolidated_summary_path}")
    
    # Print consolidated summary
    logger.info("=" * 80)
    logger.info("CONSOLIDATED BATTLE RESULTS SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Agent1: {agent1.name}")
    logger.info(f"Agent2: {agent2.name}")
    logger.info(f"Games per type: {config.NUM_EVAL_GAMES}")
    logger.info(f"Total games played: {total_games_played}")
    logger.info("-" * 80)
    
    # Calculate overall statistics
    total_agent1_wins = 0
    total_agent2_wins = 0
    total_draws = 0
    total_forfeits = 0
    
    for game_name, results in all_results.items():
        if "error" in results:
            logger.info(f"{game_name:>15}: ERROR - {results['error']}")
            continue
            
        if isinstance(results, dict):
            agent1_wins = results.get('wins_agent1', 0)
            agent2_wins = results.get('wins_agent2', 0)
            draws = results.get('draws', 0)
            valid_games_this_game = agent1_wins + agent2_wins + draws
            
            if valid_games_this_game > 0:
                agent1_rate = (agent1_wins / valid_games_this_game) * 100
                agent1_ci = calculate_wilson_ci(agent1_wins, valid_games_this_game)
                logger.info(f"{game_name:>15}: Agent1 {agent1_rate:.1f}% ({agent1_ci[0]:.1f}%-{agent1_ci[1]:.1f}%), Agent2 {100-agent1_rate-draws/valid_games_this_game*100:.1f}%, Draws {draws/valid_games_this_game*100:.1f}%")
            else:
                logger.info(f"{game_name:>15}: {results}")
        else:
            logger.info(f"{game_name:>15}: {results}")
        
        if isinstance(results, dict):
            total_agent1_wins += results.get('wins_agent1', 0)
            total_agent2_wins += results.get('wins_agent2', 0)
            total_draws += results.get('draws', 0)
            total_forfeits += results.get('forfeits_agent1', 0) + results.get('forfeits_agent2', 0)
    
    logger.info("-" * 80)
    logger.info("OVERALL TOTALS:")
    logger.info(f"Agent1 ({agent1.name}) wins: {total_agent1_wins}")
    logger.info(f"Agent2 ({agent2.name}) wins: {total_agent2_wins}")
    logger.info(f"Draws: {total_draws}")
    logger.info(f"Forfeits: {total_forfeits}")
    
    valid_games = total_agent1_wins + total_agent2_wins + total_draws
    if valid_games > 0:
        agent1_win_rate = (total_agent1_wins / valid_games) * 100
        agent2_win_rate = (total_agent2_wins / valid_games) * 100
        draw_rate = (total_draws / valid_games) * 100
        
        agent1_ci = calculate_wilson_ci(total_agent1_wins, valid_games)
        agent2_ci = calculate_wilson_ci(total_agent2_wins, valid_games)
        draw_ci = calculate_wilson_ci(total_draws, valid_games)
        
        logger.info(f"Agent1 win rate: {agent1_win_rate:.1f}% (95% CI: {agent1_ci[0]:.1f}%-{agent1_ci[1]:.1f}%)")
        logger.info(f"Agent2 win rate: {agent2_win_rate:.1f}% (95% CI: {agent2_ci[0]:.1f}%-{agent2_ci[1]:.1f}%)")
        logger.info(f"Draw rate: {draw_rate:.1f}% (95% CI: {draw_ci[0]:.1f}%-{draw_ci[1]:.1f}%)")
    
    logger.info("=" * 80)
    logger.info("Multi-game battle evaluation complete!")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())