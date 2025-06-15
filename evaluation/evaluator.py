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
        self.forfeit_by_agent_tag: Optional[str] = None # 'agent1' or 'agent2'
        self.retry_count = 0
        
    def get_current_agent_info(self):
        """Get the current agent, their player value, and an agent tag ('agent1' or 'agent2')"""
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

class EvaluationLogger:
    """Consumes events from game execution and logs them."""
    def __init__(self, agent1_name: str, agent2_name: str, game_class_name: str, config: Config):
        self.agent1_name = agent1_name
        self.agent2_name = agent2_name
        self.game_class_name = game_class_name
        self.config = config
        self.game_logs_data = {
            "agent1": agent1_name,
            "agent2": agent2_name,
            "game": game_class_name,
            "games": {}
        }
        self.results_summary = {
            "wins_agent1": 0, "wins_agent2": 0, "draws": 0,
            "forfeits_agent1": 0, "forfeits_agent2": 0
        }

    def log_game_initialization(self, game_id: int, agent1_player_value: int, agent2_player_value: int):
        self.game_logs_data["games"][str(game_id)] = {
            "agent1_player_value": agent1_player_value,
            "agent2_player_value": agent2_player_value,
            "moves": [],
            "outcome": None,
            "final_board": None
        }
        logging.debug(f"Logger: Initialized game {game_id}: {self.agent1_name}({agent1_player_value}) vs {self.agent2_name}({agent2_player_value})")

    def log_move_attempt(self, game_id: int, agent_tag: str, board_before: Any, move_input: Any, 
                         parsed_move: Optional[Any], is_valid: bool, retry_count: int, 
                         result_status: str, # "continued", "retry", "forfeit"
                         raw_output: str):
        self.game_logs_data["games"][str(game_id)]["moves"].append({
            "agent": agent_tag, # 'agent1' or 'agent2'
            "board_before": board_before,
            "move": str(parsed_move) if parsed_move is not None else str(move_input),
            "valid": is_valid,
            "retry_count": retry_count,
            "result": result_status,
            "raw_output": raw_output
        })

    def log_game_completion(self, game_id: int, final_board: Any, game_state: GameState):
        self.game_logs_data["games"][str(game_id)]["final_board"] = final_board
        outcome_str = "unknown"
        
        if game_state.forfeit_by_agent_tag:
            outcome_str = f"{game_state.forfeit_by_agent_tag}_forfeit"
            if game_state.forfeit_by_agent_tag == 'agent1':
                self.results_summary['forfeits_agent1'] += 1
            else:
                self.results_summary['forfeits_agent2'] += 1
            logging.debug(f"Logger: Game {game_id}: {game_state.forfeit_by_agent_tag} forfeited")
        else:
            winner = game_state.game.check_winner()
            if winner == game_state.agent1_player_value:
                outcome_str = "agent1_win"
                self.results_summary['wins_agent1'] += 1
                logging.debug(f"Logger: Game {game_id}: Agent1 ({self.agent1_name}) won")
            elif winner == game_state.agent2_player_value:
                outcome_str = "agent2_win"
                self.results_summary['wins_agent2'] += 1
                logging.debug(f"Logger: Game {game_id}: Agent2 ({self.agent2_name}) won")
            elif winner == 0:
                outcome_str = "draw"
                self.results_summary['draws'] += 1
                logging.debug(f"Logger: Game {game_id}: Game ended in a draw")
            else: # Should not happen if game is over
                outcome_str = "unknown_winner_state"
                logging.error(f"Logger: Game {game_id}: Unknown winner state: {winner}")


        self.game_logs_data["games"][str(game_id)]["outcome"] = outcome_str

    def generate_final_summary(self, num_total_games: int) -> Dict:
        summary = self.results_summary.copy()
        total_completed_non_forfeit = num_total_games - summary['forfeits_agent1'] - summary['forfeits_agent2']
        
        if total_completed_non_forfeit > 0:
            summary['win_rate_agent1'] = summary['wins_agent1'] / total_completed_non_forfeit
            summary['win_rate_agent2'] = summary['wins_agent2'] / total_completed_non_forfeit
            summary['draw_rate'] = summary['draws'] / total_completed_non_forfeit
        else:
            summary['win_rate_agent1'] = 0
            summary['win_rate_agent2'] = 0
            summary['draw_rate'] = 0
            
        summary['agent1_name'] = self.agent1_name
        summary['agent2_name'] = self.agent2_name
        summary['game_name'] = self.game_class_name
        summary['total_games'] = num_total_games
        return summary

    def save_detailed_logs_to_file(self) -> str:
        log_dir = os.path.join(self.config.OUTPUT_DIR, "game_logs")
        os.makedirs(log_dir, exist_ok=True)
        log_filename = f"{self.agent1_name}_vs_{self.agent2_name}_{self.game_class_name}_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
        game_log_path = os.path.join(log_dir, log_filename)
        with open(game_log_path, 'w') as f:
            json.dump(self.game_logs_data, f, indent=2)
        logging.info(f"Detailed game logs saved to {game_log_path}")
        return game_log_path

    def save_summary_report_to_file(self, results: Dict) -> str:
        output_dir = self.config.OUTPUT_DIR
        os.makedirs(output_dir, exist_ok=True)
        results_path = os.path.join(output_dir, f"evaluation_summary_{self.agent1_name}_vs_{self.agent2_name}_{datetime.now().strftime('%Y%m%d-%H%M%S')}.txt")
        
        with open(results_path, 'w') as f:
            f.write("--- Game Evaluation Results ---\n")
            f.write(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            agent1_name = results['agent1_name']
            agent2_name = results['agent2_name']
            game_name = results['game_name']
            
            f.write(f"=== {agent1_name} vs {agent2_name} on {game_name} ===\n")
            f.write(f"Total Games: {results['total_games']}\n")
            f.write(f"{agent1_name} Wins: {results['wins_agent1']} ({results.get('win_rate_agent1', 0):.2%})\n")
            f.write(f"{agent2_name} Wins: {results['wins_agent2']} ({results.get('win_rate_agent2', 0):.2%})\n")
            f.write(f"Draws: {results['draws']} ({results.get('draw_rate', 0):.2%})\n")
            f.write(f"{agent1_name} Forfeits: {results['forfeits_agent1']}\n")
            f.write(f"{agent2_name} Forfeits: {results['forfeits_agent2']}\n\n")
            
            total_completed = results['total_games'] - results['forfeits_agent1'] - results['forfeits_agent2']
            if total_completed > 0:
                f.write(f"Statistics (excluding forfeits):\n")
                f.write(f"{agent1_name} Win Rate: {results.get('win_rate_agent1',0):.2%}\n")
                f.write(f"{agent2_name} Win Rate: {results.get('win_rate_agent2',0):.2%}\n")
                f.write(f"Draw Rate: {results.get('draw_rate',0):.2%}\n")
            
            f.write("\n" + "-"*50 + "\n\n")
        
        logging.info(f"Summary report saved to {results_path}")
        return results_path

class GameRunner:
    """Manages the state and execution of multiple concurrent games."""
    def __init__(self, agent1, agent2, game_class, num_games: int, retry_limit: int, config: Config):
        self.agent1 = agent1
        self.agent2 = agent2
        self.game_class = game_class
        self.num_games = num_games
        self.retry_limit = retry_limit
        self.config = config # Assumed to have MAX_CONCURRENT_GAMES attribute
        self.game_states: List[GameState] = []
        self._initialize_games()

    def _initialize_games(self):
        """Initializes all game states."""
        for game_id in range(self.num_games):
            new_game = self.game_class()
            if random.choice([True, False]):
                agent1_player_value, agent2_player_value = 1, -1
            else:
                agent1_player_value, agent2_player_value = -1, 1
            
            gs = GameState(game_id, new_game, self.agent1, self.agent2, agent1_player_value, agent2_player_value)
            self.game_states.append(gs)
            # Event for logger would be generated by Evaluator after this
            # logging.debug(f"GameRunner: Initialized game {game_id}")

    def get_initialization_data_for_logging(self) -> List[Dict]:
        """Returns data for each game's initialization, for logging purposes."""
        init_data = []
        for gs in self.game_states:
            init_data.append({
                "game_id": gs.game_id,
                "agent1_player_value": gs.agent1_player_value,
                "agent2_player_value": gs.agent2_player_value
            })
        return init_data

    def process_one_turn(self) -> List[Dict]:
        """Processes one turn for a single selected agent across a batch of their games,
           up to MAX_CONCURRENT_GAMES. Returns list of events for logging."""
        events_for_logger = []
        active_game_states = [gs for gs in self.game_states if not gs.is_complete]

        agent_to_process_this_turn: Optional[Any] = None
        # Find the first active game and identify its current agent. This will be our target agent for this turn.
        for gs_check in active_game_states:
            if gs_check.game.is_game_over():
                gs_check.is_complete = True
            
            current_agent, _, _ = gs_check.get_current_agent_info()
            agent_to_process_this_turn = current_agent
            break 

        if not agent_to_process_this_turn:
            return events_for_logger # Should be empty if all games were completed above

        contexts_for_selected_agent: List[Dict] = []
        for gs in active_game_states: # Iterate again over potentially updated active_game_states
            if len(contexts_for_selected_agent) >= self.config.MAX_CONCURRENT_GAMES:
                logging.debug(f"GameRunner: Reached MAX_CONCURRENT_GAMES ({self.config.MAX_CONCURRENT_GAMES}) for {agent_to_process_this_turn.name}")
                break

            current_agent_in_gs, player_value, agent_tag = gs.get_current_agent_info()
            
            if current_agent_in_gs == agent_to_process_this_turn:
                contexts_for_selected_agent.append({
                    'game': gs.game, 'player_value': player_value, 'game_id': gs.game_id,
                    'game_state_obj': gs, 'agent_tag': agent_tag,
                    'agent_display_name': agent_to_process_this_turn.name
                })
        
        if not contexts_for_selected_agent:
            logging.debug(f"GameRunner: No active, non-completed games found for {agent_to_process_this_turn.name} in this pass.")
            return events_for_logger 

        logging.debug(f"GameRunner: Getting moves for {len(contexts_for_selected_agent)} games from {agent_to_process_this_turn.name}")
        
        moves = agent_to_process_this_turn.get_batch_moves(contexts_for_selected_agent)

        for idx, move_input in enumerate(moves):
            context = contexts_for_selected_agent[idx]
            gs: GameState = context['game_state_obj']
            agent_tag: str = context['agent_tag']
            agent_display_name: str = context['agent_display_name']

            if gs.is_complete: # Check again, game might have completed due to external factors or other logic not present here
                continue
            
            move_event = self._apply_move_to_game(gs, move_input, agent_tag, agent_display_name)
            events_for_logger.append(move_event) # Only move_attempt events now

            if gs.game.is_game_over(): # If move resulted in game over
                gs.is_complete = True
                # game_completion event will be logged by Evaluator
        
        return events_for_logger

    def _apply_move_to_game(self, game_state: GameState, move_input: Any, agent_tag: str, agent_display_name: str) -> Dict:
        """Applies a move to a game and returns a move event dictionary for logging."""
        game = game_state.game
        board_before = game.get_state_representation()
        parsed_move = None
        is_valid_move = False
        result_status = "continued" # "retry", "forfeit", "continued"

        if move_input is None:
            game_state.increment_retry_count()
            logging.debug(f"GameRunner: Game {game_state.game_id}: {agent_display_name} made invalid move (None), retry {game_state.retry_count}/{self.retry_limit}")
            if game_state.retry_count >= self.retry_limit:
                logging.warning(f"GameRunner: Game {game_state.game_id}: {agent_display_name} exceeded retry limit, forfeiting.")
                game.force_forfeit() # Game itself handles who forfeits based on current player
                game_state.forfeit_by_agent_tag = agent_tag
                game_state.is_complete = True
                result_status = "forfeit"
            else:
                result_status = "retry"
        else:
            parsed_move = game.parse_move_from_output(str(move_input), game.get_legal_moves())
            if parsed_move is not None:
                is_valid_move = game.make_move(parsed_move) # This applies the move
            
            if is_valid_move:
                game_state.reset_retry_count()
                logging.debug(f"GameRunner: Game {game_state.game_id}: {agent_display_name} successfully made move {parsed_move}")
                result_status = "continued"
            else: # Invalid move or failed parse
                game_state.increment_retry_count()
                logging.debug(f"GameRunner: Game {game_state.game_id}: {agent_display_name} made invalid move {move_input}, retry {game_state.retry_count}/{self.retry_limit}")
                if game_state.retry_count >= self.retry_limit:
                    logging.warning(f"GameRunner: Game {game_state.game_id}: {agent_display_name} exceeded retry limit, forfeiting.")
                    game.force_forfeit()
                    game_state.forfeit_by_agent_tag = agent_tag
                    game_state.is_complete = True
                    result_status = "forfeit"
                else:
                    result_status = "retry"
        
        return {
            "type": "move_attempt", "game_id": game_state.game_id, "agent_tag": agent_tag,
            "board_before": board_before, "move_input": move_input, "parsed_move": parsed_move,
            "is_valid": is_valid_move, "retry_count": game_state.retry_count if result_status != "forfeit" else self.retry_limit,
            "result_status": result_status, "raw_output": str(move_input)
        }

    def get_completed_game_count(self) -> int:
        return sum(1 for gs in self.game_states if gs.is_complete)

class Evaluator:
    """High-level orchestrator for evaluating agents."""
    
    def __init__(self, config: Optional[Config] = None, retry_limit: int = 3):
        self.config = config or Config()
        self.retry_limit = retry_limit
        # self.results and self.game_logs are now managed by EvaluationLogger
        
    def evaluate_agent_vs_agent(self, agent1, agent2, game_class, num_games: Optional[int] = None, no_logging: bool = False) -> Dict:
        """
        Evaluate one agent against another on a specific game.
        Orchestrates GameRunner and EvaluationLogger.
        """
        num_games_to_run = num_games if num_games is not None else self.config.NUM_EVAL_GAMES
        
        logging.info(f"Starting evaluation: {agent1.name} vs {agent2.name} on {game_class.__name__} for {num_games_to_run} games.")

        logger = EvaluationLogger(agent1.name, agent2.name, game_class.__name__, self.config)
        runner = GameRunner(agent1, agent2, game_class, num_games_to_run, self.retry_limit, self.config)

        # Log initial game states
        initial_game_data = runner.get_initialization_data_for_logging()
        for init_data in initial_game_data:
            logger.log_game_initialization(
                game_id=init_data["game_id"],
                agent1_player_value=init_data["agent1_player_value"],
                agent2_player_value=init_data["agent2_player_value"]
            )
            
        pbar = tqdm(total=num_games_to_run, desc=f"Eval: {agent1.name} vs {agent2.name}")
        
        completed_count = 0
        while completed_count < num_games_to_run:
            events = runner.process_one_turn() # Now only returns move_attempt events
            
            for event in events:
                if event["type"] == "move_attempt": # This will always be true now
                    logger.log_move_attempt(
                        game_id=event["game_id"], agent_tag=event["agent_tag"],
                        board_before=event["board_before"], move_input=event["move_input"],
                        parsed_move=event["parsed_move"], is_valid=event["is_valid"],
                        retry_count=event["retry_count"], result_status=event["result_status"],
                        raw_output=event["raw_output"]
                    )

            current_total_completed = runner.get_completed_game_count()
            newly_completed_this_turn = current_total_completed - completed_count
            if newly_completed_this_turn > 0:
                pbar.update(newly_completed_this_turn)
                completed_count = current_total_completed
            
            if not events and completed_count < num_games_to_run:
                all_games_done = all(gs.is_complete for gs in runner.game_states)
                if all_games_done:
                    logging.warning("Evaluator: All games marked complete by runner, but loop condition not met. Updating progress and breaking.")
                    if pbar.n < num_games_to_run:
                         pbar.update(num_games_to_run - pbar.n)
                    break 
                time.sleep(0.001) 
            elif completed_count >= num_games_to_run:
                break

        pbar.close()

        # Log all game completions after the main loop
        for gs in runner.game_states:
            if gs.is_complete: # Ensure we only log completed games
                logger.log_game_completion(
                    game_id=gs.game_id,
                    final_board=gs.game.get_state_representation(), # Get final board state here
                    game_state=gs
                )
        
        final_summary_results = logger.generate_final_summary(num_games_to_run)
        if not no_logging:
            logger.save_detailed_logs_to_file()
            logger.save_summary_report_to_file(final_summary_results)
        
        logging.info(f"Evaluation finished for {agent1.name} vs {agent2.name}.")
        return final_summary_results
