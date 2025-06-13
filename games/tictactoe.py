import copy
from typing import List, Tuple, Optional, Any
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
        self.board = [0] * 9  # 0: empty, 1: Player X, -1: Player O
        self.current_player = 1  # Player X starts
        self.player_X_symbol = 'X'
        self.player_O_symbol = 'O'
        self.empty_symbol = '.'
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
            "Your response MUST be your chosen (row,col) coordinate. "
            "Example: If you want to play on row 1, column 2, your output should be `(1,2)`. "
            "Do not add any other text or explanation."
        )
        self.user_prompt_template_no_thinking = (
            "{board_representation}\n"
            "You are player '{player_symbol}'.\n"
            "Your available legal moves (row,col format): [{legal_moves_str}]\n"
            "Choose your move by selecting one of the available (row,col) pairs. Your move (e.g., `(1,2)` for row 1, column 2):"
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
    
    def get_state_representation(self) -> str:
        """Get string representation of current game state"""
        board_str = "Current Board State:\n"
        for r in range(3):
            row_cells = []
            for c in range(3):
                idx = r * 3 + c
                symbol = self.get_player_symbol(self.board[idx])
                row_cells.append(f"({r},{c}):\"{symbol}\"")
            board_str += "  ".join(row_cells) + "\n"
        
        current_player_symbol = self.get_player_symbol(self.current_player)
        board_str += f"Current turn: {current_player_symbol} (plays as {self.current_player})\n"
        board_str += f"Legal moves (row,col): {self.get_legal_moves()}\n"
        
        return board_str
    
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
    
    def get_chat_history_for_llm(self, llm: Agent) -> str:
        """Get prompt for agent describing current game state"""
        board_representation = self.get_board_representation_for_llm()
        player_symbol = self.get_player_symbol(self.current_player)
        legal_moves = self.get_legal_moves()
        legal_moves_str = ", ".join([f"({r},{c})" for r, c in legal_moves])
        
        system_prompt = self.system_prompt
        user_prompt_template = self.user_prompt_template
        
        if isinstance(llm, VLLMAgent) and not llm.enable_thinking:
            system_prompt = self.system_prompt_no_thinking
            user_prompt = self.user_prompt_template_no_thinking
            
        user_prompt = user_prompt_template.format(
            board_representation=board_representation,
            player_symbol=player_symbol,
            legal_moves_str=legal_moves_str
        )
        
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    
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
        
    def clone(self):
        """Create a deep copy of the game"""
        return copy.deepcopy(self)