import os
import time
import random
import logging
import json
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from config import Config

class GameState:
    """Represents the state of a single game during evaluation"""
    
    def __init__(self, game_id: int, game_instance, agent1, agent2, agent1_player_value: int, agent2_player_value: int):
        self.game_id = game_id
        self.game = game_instance
        self.agent1 = agent1
        self.agent2 = agent2
        self.agent1_player_value = agent1_player_value
        self.agent2_player_value = agent2_player_value
        self.is_complete = False
        self.forfeit_by = None
        self.retry_count = 0
        
    def get_current_agent_info(self):
        """Get the current agent and their player value"""
        current_player = self.game.get_current_player()
        if current_player == self.agent1_player_value:
            return self.agent1, self.agent1_player_value, 'agent1'
        else:
            return self.agent2, self.agent2_player_value, 'agent2'
    
    def get_retry_count(self) -> int:
        """Get retry count"""
        return self.retry_count
    
    def increment_retry_count(self):
        """Increment retry count"""
        self.retry_count += 1
    
    def reset_retry_count(self):
        """Reset retry count"""
        self.retry_count = 0

class Evaluator:
    """Handles evaluation of agents on games"""
    
    def __init__(self, config=None, retry_limit=3):
        self.config = config or Config()
        self.retry_limit = retry_limit
        self.results = {}
        self.game_logs = {}  
        
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
        
        # Initialize logging structure
        log_dir = os.path.join(self.config.OUTPUT_DIR, "game_logs")
        os.makedirs(log_dir, exist_ok=True)
        log_filename = f"{agent1.name}_vs_{agent2.name}_{game_class.__name__}_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
        
        self.game_logs = {
            "agent1": agent1.name,
            "agent2": agent2.name,
            "game": game_class.__name__,
            "games": {}
        }
        
        # Initialize all games at start
        game_states = self._initialize_all_games(num_games, game_class, agent1, agent2)
        
        # Process games in batches until all are complete
        pbar = tqdm(total=num_games, desc=f"{agent1.name} vs {agent2.name}")
        games_completed = 0
        
        while games_completed < num_games:
            # Get active (incomplete) games
            active_games = [gs for gs in game_states if not gs.is_complete]
            
            if not active_games:
                break
                
            # Process one batch of moves for active games
            newly_completed = self._process_game_batch(active_games)
            games_completed += newly_completed
            pbar.update(newly_completed)
            
            # Small delay to prevent busy waiting
            if not newly_completed and active_games:
                time.sleep(0.001)
        
        pbar.close()
        
        # Count final results from all games
        results = self._count_final_results(game_states, agent1, agent2, game_class, num_games)
        
        # Save game logs
        game_log_path = os.path.join(log_dir, log_filename)
        with open(game_log_path, 'w') as f:
            json.dump(self.game_logs, f, indent=2)
        logging.info(f"Game logs saved to {game_log_path}")
        
        return results
    
    def _initialize_all_games(self, num_games: int, game_class, agent1, agent2) -> List[GameState]:
        """Initialize all games at the start"""
        game_states = []
        
        for game_id in range(num_games):
            new_game = game_class()
            
            # Randomly assign player values
            if random.choice([True, False]):
                agent1_player_value = 1
                agent2_player_value = -1
            else:
                agent1_player_value = -1
                agent2_player_value = 1
            
            game_state = GameState(
                game_id=game_id,
                game_instance=new_game,
                agent1=agent1,
                agent2=agent2,
                agent1_player_value=agent1_player_value,
                agent2_player_value=agent2_player_value
            )
            
            game_states.append(game_state)
            
            # Initialize game log entry
            self.game_logs["games"][str(game_id)] = {
                "agent1_player_value": agent1_player_value,
                "agent2_player_value": agent2_player_value,
                "moves": [],
                "outcome": None
            }
            
            logging.debug(f"Initialized game {game_id}: {agent1.name}({agent1_player_value}) vs {agent2.name}({agent2_player_value})")
        
        return game_states
    
    def _process_game_batch(self, active_games: List[GameState]) -> int:
        """Process one batch of moves for active games, returns number of newly completed games"""
        if not active_games:
            return 0
        
        # Group games by current agent
        agent_contexts = {}  # agent -> list of contexts
        
        for game_state in active_games:
            if game_state.game.is_game_over():
                game_state.is_complete = True
                continue
                
            current_agent, player_value, agent_name = game_state.get_current_agent_info()
            
            if current_agent not in agent_contexts:
                agent_contexts[current_agent] = []
            
            agent_contexts[current_agent].append({
                'game': game_state.game,
                'player_value': player_value,
                'game_id': game_state.game_id,
                'game_state': game_state,
                'agent_name': agent_name
            })
        
        # Get moves from each agent for their respective games
        for agent, contexts in agent_contexts.items():
            if not contexts:
                continue
                
            logging.debug(f"Getting moves for {len(contexts)} games from {agent.name}")
            moves = agent.get_batch_moves(contexts)
            
            # Apply moves
            for idx, move in enumerate(moves):
                context = contexts[idx]
                game_state = context['game_state']
                agent_name = context['agent_name']
                
                if game_state.game.is_game_over():
                    continue
                
                self._apply_move_and_log(game_state, move, agent_name, agent.name)
        
        # Count newly completed games
        newly_completed = 0
        for game_state in active_games:
            if not game_state.is_complete and game_state.game.is_game_over():
                game_state.is_complete = True
                self._finalize_game_log(game_state)
                newly_completed += 1
        
        return newly_completed
    
    def _apply_move_and_log(self, game_state: GameState, move, agent_name: str, agent_display_name: str):
        """Apply a move and log the result with retry logic"""
        game_id = game_state.game_id
        game = game_state.game
        
        # Record current board state
        board_state = game.get_state_representation()
        
        if move is None:
            # Invalid move - check retry limit
            current_retries = game_state.retry_count
            game_state.increment_retry_count()
            
            logging.debug(f"Game {game_id}: {agent_display_name} made invalid move (None), retry {current_retries + 1}/{self.retry_limit}")
            
            self.game_logs["games"][str(game_id)]["moves"].append({
                "agent": agent_name,
                "board_before": board_state,
                "move": None,
                "valid": False,
                "retry_count": current_retries + 1,
                "result": "retry" if current_retries + 1 < self.retry_limit else "forfeit",
                "raw_output": str(move)
            })
            
            if current_retries + 1 >= self.retry_limit:
                # Exceeded retry limit - forfeit
                logging.warning(f"Game {game_id}: {agent_display_name} exceeded retry limit ({self.retry_limit}), forfeiting")
                game.force_forfeit()
                game_state.forfeit_by = agent_name
                game_state.is_complete = True
        else:
            # Parse and apply move
            logging.debug(f"Game {game_id}: {agent_display_name} attempting move {move}")
            parsed_move = game.parse_move_from_output(str(move), game.get_legal_moves())
            success = False
            if parsed_move is not None:
                success = game.make_move(parsed_move)
            
            if not success:
                # Invalid move - check retry limit
                current_retries = game_state.retry_count
                game_state.increment_retry_count()
                
                logging.debug(f"Game {game_id}: {agent_display_name} made invalid move {move}, retry {current_retries + 1}/{self.retry_limit}")
                
                self.game_logs["games"][str(game_id)]["moves"].append({
                    "agent": agent_name,
                    "board_before": board_state,
                    "move": str(parsed_move) if parsed_move is not None else str(move),
                    "valid": False,
                    "retry_count": current_retries + 1,
                    "result": "retry" if current_retries + 1 < self.retry_limit else "forfeit",
                    "raw_output": str(move)
                })
                
                if current_retries + 1 >= self.retry_limit:
                    # Exceeded retry limit - forfeit
                    logging.warning(f"Game {game_id}: {agent_display_name} exceeded retry limit ({self.retry_limit}), forfeiting")
                    game.force_forfeit()
                    game_state.forfeit_by = agent_name
                    game_state.is_complete = True
            else:
                # Valid move - reset retry counter and log success
                game_state.reset_retry_count()
                
                logging.debug(f"Game {game_id}: {agent_display_name} successfully made move {move}")
                
                self.game_logs["games"][str(game_id)]["moves"].append({
                    "agent": agent_name,
                    "board_before": board_state,
                    "move": str(parsed_move),
                    "valid": True,
                    "retry_count": 0,
                    "result": "continued",
                    "raw_output": str(move)
                })
    
    def _finalize_game_log(self, game_state: GameState):
        """Finalize the game log with outcome"""
        game_id = game_state.game_id
        game = game_state.game
        
        # Record final board state
        final_board = game.get_state_representation()
        self.game_logs["games"][str(game_id)]["final_board"] = final_board
        
        # Determine outcome
        outcome = None
        if game_state.forfeit_by:
            outcome = f"{game_state.forfeit_by}_forfeit"
            logging.debug(f"Game {game_id}: {game_state.forfeit_by} forfeited")
        else:
            winner = game.check_winner()
            if winner == game_state.agent1_player_value:
                outcome = "agent1_win"
                logging.debug(f"Game {game_id}: Agent1 ({game_state.agent1.name}) won")
            elif winner == game_state.agent2_player_value:
                outcome = "agent2_win"
                logging.debug(f"Game {game_id}: Agent2 ({game_state.agent2.name}) won")
            elif winner == 0:
                outcome = "draw"
                logging.debug(f"Game {game_id}: Game ended in a draw")
        
        self.game_logs["games"][str(game_id)]["outcome"] = outcome
    
    def _count_final_results(self, game_states: List[GameState], agent1, agent2, game_class, num_games: int) -> Dict:
        """Count final results from all completed games"""
        results = {
            "wins_agent1": 0, 
            "wins_agent2": 0, 
            "draws": 0, 
            "forfeits_agent1": 0, 
            "forfeits_agent2": 0
        }
        
        for game_state in game_states:
            if game_state.forfeit_by == 'agent1':
                results['forfeits_agent1'] += 1
            elif game_state.forfeit_by == 'agent2':
                results['forfeits_agent2'] += 1
            else:
                winner = game_state.game.check_winner()
                if winner == game_state.agent1_player_value:
                    results['wins_agent1'] += 1
                elif winner == game_state.agent2_player_value:
                    results['wins_agent2'] += 1
                elif winner == 0:
                    results['draws'] += 1
        
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