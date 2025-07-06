import copy
from typing import List, Tuple, Optional, Dict
import sys
import os
import re
import logging
from agents.agent import Agent
from agents.vllm_agent import VLLMAgent

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from games.game import Game

class TicTacToeGame(Game):
    """Tic-Tac-Toe game implementation"""
    
    def __init__(self):
        super().__init__()
        self.board = [0] * 9  # 0: empty, 1: Player X, -1: Player O
        self.player_X_symbol = 'X'
        self.player_O_symbol = 'O'
        self._game_over_forced_forfeit = False  # For forcing game end in evaluation
        self.prompt_template = None  # For dynamic prompt updates
        self._setup_default_prompt()
        
    def _setup_default_prompt(self):
        """Set default prompt template"""
        self.system_prompt = (
            "You are an expert Tic-Tac-Toe player. Your task is to choose the best move. "
            "First, think *briefly and concisely* about the current board state inside a `<think>` block. Your thinking should be a few sentences at most. "
            "After your thinking block, you MUST provide your chosen move as a (row,col) coordinate pair. "
            "Your entire response should follow this format: `<think>Your reasoning here...</think>(row,col). "
            "Do not add any other text outside this structure."
        )
        self.user_prompt_template = (
            "{board_representation}\n"
            "You are player '{player_symbol}'.\n"
            "Your available legal moves (row,col format): [{legal_moves_str}]\n"
            "Provide your thinking and final move in the specified format: `<think>...</think>(r,c)`"
        )
        self.system_prompt_no_thinking = (
            "You are playing Tic-Tac-Toe. Your task is to select the BEST move from the available legal moves. "
            "Your response MUST be your chosen (row,col) coordinate, enclosed in square brackets. "
            "Example: If you want to play on row 1, column 2, your output should be `[1,2]`. "
            "Do not add any other text or explanation."
        )
        self.user_prompt_template_no_thinking = (
            "{board_representation}\n"
            "You are player '{player_symbol}'.\n"
            "Your available legal moves (row,col format): [{legal_moves_str}]\n"
            "Choose your move by selecting one of the available (row,col) pairs. Your move (e.g., `[1,2]` for row 1, column 2):"
        )
    
    def update_prompt(self, system_prompt=None, user_prompt_template=None):
        """Allow updating prompt templates"""
        if system_prompt:
            self.system_prompt = system_prompt
        if user_prompt_template:
            self.user_prompt_template = user_prompt_template
    
    def get_player_symbol(self, player_value):
        """Get the symbol representation for a player"""
        if player_value == 1: 
            return self.player_X_symbol
        if player_value == -1: 
            return self.player_O_symbol
        return self.empty_symbol
    
    def get_legal_moves(self) -> List[Tuple[int, int]]:
        """Return all legal moves as (row, col) coordinate pairs"""
        return [(i // 3, i % 3) for i, spot in enumerate(self.board) if spot == 0]
    
    def make_move(self, move_coord: Tuple[int, int]) -> bool:
        """Execute a move, return True if move was legal and successful"""
        if not (isinstance(move_coord, tuple) and len(move_coord) == 2 and
                0 <= move_coord[0] < 3 and 0 <= move_coord[1] < 3):
            return False 
        row, col = move_coord
        move_idx = row * 3 + col
        if self.board[move_idx] == 0:
            self.board[move_idx] = self.current_player
            self.current_player *= -1  # Switch player
            return True
        return False
    
    def check_winner(self) -> Optional[int]:
        """Check for a winner, return 1 (X wins), -1 (O wins), 0 (draw), None (game continues)"""
        lines = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Rows
            [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Columns
            [0, 4, 8], [2, 4, 6]              # Diagonals
        ]
        for line in lines:
            s = sum(self.board[i] for i in line)
            if s == 3: 
                return 1  # X wins
            if s == -3: 
                return -1  # O wins
        
        if 0 not in self.board: 
            return 0  # Board is full, draw
        
        return None  # Game continues
    
    def is_game_over(self) -> bool:
        """Check if the game is over"""
        return self.check_winner() is not None or self._game_over_forced_forfeit
    
    def get_current_player(self) -> int:
        """Get current player (1 for X, -1 for O)"""
        return self.current_player
    
    def get_board_representation_for_llm(self) -> str:
        """Get board state representation for LLM"""
        s = "Current board state (row,col format):\n"
        for r in range(3):
            row_str_parts = []
            for c in range(3):
                idx = r * 3 + c
                symbol = self.get_player_symbol(self.board[idx])
                row_str_parts.append(f"({r},{c}):\"{symbol}\"")
            s += "  ".join(row_str_parts) + "\n" 
        return s.strip()
    
    def get_chat_history_for_llm(self, llm: Agent) -> List[Dict[str, str]]:
        """Get prompt for agent describing current game state"""
        return super().get_chat_history_for_llm(llm)
    
    def _format_legal_moves_for_prompt(self, legal_moves: List[Tuple[int, int]]) -> str:
        """Format legal moves for TicTacToe as (row,col) pairs"""
        return ", ".join([f"({r},{c})" for r, c in legal_moves])
    
    def parse_move_from_output(self, raw_output: str, legal_moves: List[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
        """
        Parse move from LLM output for Tic-Tac-Toe
        
        Args:
            raw_output: Raw text output from the LLM
            legal_moves: List of legal moves for validation
            
        Returns:
            Parsed move as (row, col) tuple or None if no valid move found
        """
        # Look for coordinates in the format (row,col)
        match = re.findall(r'\(\s*([0-2])\s*,\s*([0-2])\s*\)', raw_output)
        if not match:
            match = re.findall(r'\[\s*([0-2])\s*,\s*([0-2])\s*\]', raw_output)
        if match:
            last_match = match[-1]
            move_coord = (int(last_match[0]), int(last_match[1]))
            if move_coord in legal_moves:
                return move_coord
            else:
                logging.warning(f"LLM chose illegal move {move_coord} (not in {legal_moves}). Raw: '{raw_output}'")
                return None
        else:
            logging.warning(f"LLM output format incorrect: '{raw_output}'")
            return None
    
    def reset(self) -> None:
        """Reset game to initial state"""
        self.board = [0] * 9
        self.current_player = 1
        self._game_over_forced_forfeit = False
    
    def force_forfeit(self) -> None:
        """Force game end (for evaluation)"""
        self._game_over_forced_forfeit = True
        
    def evaluate_position(self) -> float:
        """
        Evaluate the current position from player 1's perspective.
        For TicTacToe, since games are short, this is mainly for completeness.
        """
        # For TicTacToe, positions are usually evaluated to completion
        # This is mainly for edge cases or when depth is artificially limited
        winner = self.check_winner()
        if winner is not None:
            if winner == 1:
                return 10.0  # X wins
            elif winner == -1:
                return -10.0  # O wins
            else:
                return 0.0  # Draw
        
        # For non-terminal positions, evaluate based on potential winning lines
        score = 0.0
        lines = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Rows
            [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Columns
            [0, 4, 8], [2, 4, 6]              # Diagonals
        ]
        
        for line in lines:
            x_count = sum(1 for i in line if self.board[i] == 1)
            o_count = sum(1 for i in line if self.board[i] == -1)
            empty_count = sum(1 for i in line if self.board[i] == 0)
            
            # If both players have pieces in the line, it's blocked
            if x_count > 0 and o_count > 0:
                continue
                
            # Score for potential winning lines
            if x_count == 2 and empty_count == 1:
                score += 5.0
            elif x_count == 1 and empty_count == 2:
                score += 1.0
            elif o_count == 2 and empty_count == 1:
                score -= 5.0
            elif o_count == 1 and empty_count == 2:
                score -= 1.0
        
        return score
    
    def clone(self):
        """Create a deep copy of the game"""
        return copy.deepcopy(self)
    
    def get_action_rewards(self) -> Dict[str, float]:
        '''Use minimax agent to get rewards for each action'''
        from agents.minimax_agent_tictactoe import MinimaxAgentTicTacToe
        if not hasattr(TicTacToeGame, '_minimax_agent'):
            TicTacToeGame._minimax_agent = MinimaxAgentTicTacToe()
        return TicTacToeGame._minimax_agent.get_action_rewards(self)
    
    def load_state_from_representation(self, state_str: str) -> bool:
        """
        Load TicTacToe game state from string representation.
        
        Args:
            state_str: String representation from get_state_representation()
            
        Returns:
            bool: True if state was loaded successfully, False if parsing failed
        """
        try:
            lines = state_str.strip().split('\n')
            
            # Check if this looks like a valid TicTacToe state
            if not any("row,col format" in line for line in lines):
                return False
            
            # Parse board from the coordinate format
            new_board = [0] * 9
            found_coordinates = 0
            
            for line in lines:
                if line.strip().startswith("Current turn:"):
                    break
                
                # Look for coordinate patterns like (0,0):"X"
                import re
                matches = re.findall(r'\((\d),(\d)\):"([XO.])"', line)
                for match in matches:
                    row, col, symbol = int(match[0]), int(match[1]), match[2]
                    if row >= 3 or col >= 3:  # Invalid coordinates
                        return False
                    idx = row * 3 + col
                    if 0 <= idx < 9:
                        if symbol == 'X':
                            new_board[idx] = 1
                        elif symbol == 'O':
                            new_board[idx] = -1
                        else:  # '.'
                            new_board[idx] = 0
                        found_coordinates += 1
            
            # We should have found exactly 9 coordinates for a valid TicTacToe board
            if found_coordinates != 9:
                return False
            
            # Parse current player
            current_player = 1
            found_current_turn = False
            for line in lines:
                if line.strip().startswith("Current turn:"):
                    found_current_turn = True
                    if "(plays as -1)" in line:
                        current_player = -1
                    elif "(plays as 1)" in line:
                        current_player = 1
                    break
            
            if not found_current_turn:
                return False
            
            # Update game state
            self.board = new_board
            self.current_player = current_player
            self._game_over_forced_forfeit = False
            
            return True
            
        except Exception:
            return False