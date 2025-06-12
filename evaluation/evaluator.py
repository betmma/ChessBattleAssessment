import os
import time
import random
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from config import Config

class Evaluator:
    """Handles evaluation of agents on games"""
    
    def __init__(self, config=None):
        self.config = config or Config()
        self.results = {}
        
    def evaluate_agent_vs_agent(self, agent1, agent2, game_class, num_games=None):
        """
        Evaluate one agent against another on a specific game
        
        Args:
            agent1: First agent
            agent2: Second agent
            game_class: Game class to use for evaluation
            num_games: Number of games to run (default: use config)
            
        Returns:
            Results dictionary
        """
        if num_games is None:
            num_games = self.config.NUM_EVAL_GAMES
        
        logging.info(f"Starting evaluation: {agent1.name} vs {agent2.name} on {game_class.__name__}")
        results = {"wins_agent1": 0, "wins_agent2": 0, "draws": 0, "forfeits_agent1": 0, "forfeits_agent2": 0}
        
        # Create game instances based on the number of games
        active_games_data = []
        games_launched = 0
        games_completed = 0
        
        pbar = tqdm(total=num_games, desc=f"{agent1.name} vs {agent2.name}")
        
        # Determine if we can batch process for each agent
        can_batch_agent1 = agent1.supports_batch()
        can_batch_agent2 = agent2.supports_batch()
        
        while games_completed < num_games:
            # Phase 1: Process existing games
            agent1_contexts = []
            agent2_contexts = []
            indices_to_remove = []
            
            for i, game_data in enumerate(active_games_data):
                game = game_data['game']
                current_player = game.get_current_player()
                game_id = game_data['game_id']
                
                if game.is_game_over():
                    indices_to_remove.append(i)
                    continue
                
                if current_player == game_data['agent1_player_value']:
                    # Agent 1's turn
                    agent1_contexts.append({
                        'game': game,
                        'player_value': game_data['agent1_player_value'],
                        'game_id': game_id
                    })
                else:
                    # Agent 2's turn
                    agent2_contexts.append({
                        'game': game,
                        'player_value': game_data['agent2_player_value'],
                        'game_id': game_id
                    })
            
            # Get moves from agents (batch if supported)
            agent1_moves = agent1.get_batch_moves(agent1_contexts) if agent1_contexts else []
            agent2_moves = agent2.get_batch_moves(agent2_contexts) if agent2_contexts else []
            
            # Apply agent1 moves
            for idx, move in enumerate(agent1_moves):
                context = agent1_contexts[idx]
                game_id = context['game_id']
                game_data = next((g for g in active_games_data if g['game_id'] == game_id), None)
                
                if game_data and not game_data['game'].is_game_over():
                    if move is None:
                        # Invalid move - forfeit
                        game_data['game'].force_forfeit()
                        game_data['forfeit_by'] = 'agent1'
                        indices_to_remove.append(active_games_data.index(game_data))
                    else:
                        # Apply move
                        success = game_data['game'].make_move(move)
                        if not success:
                            # Invalid move - forfeit
                            game_data['game'].force_forfeit()
                            game_data['forfeit_by'] = 'agent1'
                            indices_to_remove.append(active_games_data.index(game_data))
            
            # Apply agent2 moves
            for idx, move in enumerate(agent2_moves):
                context = agent2_contexts[idx]
                game_id = context['game_id']
                game_data = next((g for g in active_games_data if g['game_id'] == game_id), None)
                
                if game_data and not game_data['game'].is_game_over():
                    if move is None:
                        # Invalid move - forfeit
                        game_data['game'].force_forfeit()
                        game_data['forfeit_by'] = 'agent2'
                        indices_to_remove.append(active_games_data.index(game_data))
                    else:
                        # Apply move
                        success = game_data['game'].make_move(move)
                        if not success:
                            # Invalid move - forfeit
                            game_data['game'].force_forfeit()
                            game_data['forfeit_by'] = 'agent2'
                            indices_to_remove.append(active_games_data.index(game_data))
            
            # Clean up completed games
            indices_to_remove = sorted(list(set(indices_to_remove)), reverse=True)
            for i in indices_to_remove:
                ended_game_data = active_games_data.pop(i)
                game = ended_game_data['game']
                
                if game.is_game_over():
                    games_completed += 1
                    pbar.update(1)
                    
                    # Determine outcome
                    winner = game.check_winner()
                    if 'forfeit_by' in ended_game_data:
                        if ended_game_data['forfeit_by'] == 'agent1':
                            results['forfeits_agent1'] += 1
                        else:
                            results['forfeits_agent2'] += 1
                    elif winner == ended_game_data['agent1_player_value']:
                        results['wins_agent1'] += 1
                    elif winner == ended_game_data['agent2_player_value']:
                        results['wins_agent2'] += 1
                    elif winner == 0:
                        results['draws'] += 1
            
            # Phase 2: Launch new games
            num_to_add = min(self.config.VLLM_MAX_CONCURRENT_GAMES - len(active_games_data), 
                             num_games - games_launched)
            
            for _ in range(num_to_add):
                if games_launched >= num_games:
                    break
                
                new_game = game_class()
                # Randomly decide which agent is X (1) and which is O (-1)
                if random.choice([True, False]):
                    agent1_player_value = 1
                    agent2_player_value = -1
                else:
                    agent1_player_value = -1
                    agent2_player_value = 1
                
                game_id = games_launched
                
                new_game_data = {
                    'game': new_game,
                    'agent1_player_value': agent1_player_value,
                    'agent2_player_value': agent2_player_value,
                    'game_id': game_id
                }
                
                active_games_data.append(new_game_data)
                games_launched += 1
            
            # Small sleep if no work this cycle
            if not agent1_contexts and not agent2_contexts:
                if len(active_games_data) > 0 or games_launched < num_games:
                    time.sleep(0.005)
        
        pbar.close()
        
        # Ensure total adds up to num_games
        accounted_games = (results['wins_agent1'] + results['wins_agent2'] + 
                           results['draws'] + results['forfeits_agent1'] + 
                           results['forfeits_agent2'])
        if accounted_games < num_games:
            discrepancy = num_games - accounted_games
            logging.warning(f"{discrepancy} games unaccounted for. Adjusting forfeits.")
            # Split the discrepancy between both agents
            results['forfeits_agent1'] += discrepancy // 2
            results['forfeits_agent2'] += discrepancy - (discrepancy // 2)
        
        # Calculate win rates
        total_completed = num_games - results['forfeits_agent1'] - results['forfeits_agent2']
        if total_completed > 0:
            results['win_rate_agent1'] = results['wins_agent1'] / total_completed
            results['win_rate_agent2'] = results['wins_agent2'] / total_completed
            results['draw_rate'] = results['draws'] / total_completed
        else:
            results['win_rate_agent1'] = 0
            results['win_rate_agent2'] = 0
            results['draw_rate'] = 0
        
        results['agent1_name'] = agent1.name
        results['agent2_name'] = agent2.name
        results['game_name'] = game_class.__name__
        results['total_games'] = num_games
        
        return results
    
    def save_results(self, results, output_dir=None):
        """
        Save evaluation results to a file
        
        Args:
            results: Results dictionary
            output_dir: Directory to save results (default: use config)
        """
        if output_dir is None:
            output_dir = self.config.OUTPUT_DIR
        
        os.makedirs(output_dir, exist_ok=True)
        results_path = os.path.join(output_dir, f"evaluation_summary_{datetime.now().strftime('%Y%m%d-%H%M%S')}.txt")
        
        with open(results_path, 'w') as f:
            f.write("--- Game Evaluation Results ---\n")
            f.write(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            for result in results:
                agent1_name = result['agent1_name']
                agent2_name = result['agent2_name']
                game_name = result['game_name']
                
                f.write(f"=== {agent1_name} vs {agent2_name} on {game_name} ===\n")
                f.write(f"Total Games: {result['total_games']}\n")
                f.write(f"{agent1_name} Wins: {result['wins_agent1']} ({result['win_rate_agent1']:.2%})\n")
                f.write(f"{agent2_name} Wins: {result['wins_agent2']} ({result['win_rate_agent2']:.2%})\n")
                f.write(f"Draws: {result['draws']} ({result['draw_rate']:.2%})\n")
                f.write(f"{agent1_name} Forfeits: {result['forfeits_agent1']}\n")
                f.write(f"{agent2_name} Forfeits: {result['forfeits_agent2']}\n\n")
                
                # Calculate overall statistics excluding forfeits
                total_completed = result['total_games'] - result['forfeits_agent1'] - result['forfeits_agent2']
                if total_completed > 0:
                    f.write(f"Statistics (excluding forfeits):\n")
                    f.write(f"{agent1_name} Win Rate: {result['win_rate_agent1']:.2%}\n")
                    f.write(f"{agent2_name} Win Rate: {result['win_rate_agent2']:.2%}\n")
                    f.write(f"Draw Rate: {result['draw_rate']:.2%}\n")
                
                f.write("\n" + "-"*50 + "\n\n")
        
        logging.info(f"Results saved to {results_path}")
        return results_path