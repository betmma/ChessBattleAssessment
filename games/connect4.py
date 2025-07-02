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

class Connect4Game(Game):
    """Connect 4 game implementation"""
    
    def __init__(self):
        super().__init__()
        self.rows = 6
        self.cols = 7
        self.board = [[0 for _ in range(self.cols)] for _ in range(self.rows)]
        self.player_1_symbol = 'R'  # Red
        self.player_2_symbol = 'Y'  # Yellow
        self._setup_default_prompt()
        
    def _setup_default_prompt(self):
        """Set default prompt template"""
        self.system_prompt = (
            "You are an expert Connect 4 player. Your task is to choose the best column to drop your piece. "
            "Your thinking should be a few sentences at most. "
            "After your thinking block, you MUST provide your chosen column number (0-6), enclosed in square brackets. "
        )
        self.user_prompt_template = (
            "{board_representation}\n"
            "You are player '{player_symbol}'.\n"
            "Your available legal moves (columns): [{legal_moves_str}]\n"
            "Provide your thinking and final move in the specified format: `[column_number]`"
        )
        self.system_prompt_no_thinking = (
            "You are playing Connect 4. Your task is to select the BEST column to drop your piece from the available legal moves. "
            "Your response MUST be your chosen column number (0-6), enclosed in square brackets. "
            "Example: If you want to drop in column 3, your output should be `[3]`. "
            "Do not add any other text or explanation."
        )
        self.user_prompt_template_no_thinking = (
            "{board_representation}\n"
            "You are player '{player_symbol}'.\n"
            "Your available legal moves (columns): [{legal_moves_str}]\n"
            "Choose your move by selecting one of the available column numbers. Your move (e.g., `[3]` for column 3):"
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
            return self.player_1_symbol
        if player_value == -1: 
            return self.player_2_symbol
        return self.empty_symbol
    
    def get_legal_moves(self) -> List[int]:
        """Return all legal moves as column numbers"""
        return [col for col in range(self.cols) if self.board[0][col] == 0]
    
    def make_move(self, move: int) -> bool:
        """Execute a move, return True if move was legal and successful"""
        if not isinstance(move, int) or move < 0 or move >= self.cols:
            return False
        
        # Check if column is full
        if self.board[0][move] != 0:
            return False
            
        # Find the lowest empty row in the column
        for row in range(self.rows - 1, -1, -1):
            if self.board[row][move] == 0:
                self.board[row][move] = self.current_player
                self.current_player *= -1  # Switch player (1 -> -1, -1 -> 1)
                return True
        
        return False
    
    def check_winner(self) -> Optional[int]:
        """Check for a winner, return 1 (Red wins), -1 (Yellow wins), 0 (draw), None (game continues)"""
        # Check for 4 in a row (horizontal, vertical, diagonal)
        directions = [
            (0, 1),   # Horizontal
            (1, 0),   # Vertical
            (1, 1),   # Diagonal \
            (1, -1)   # Diagonal /
        ]
        
        for row in range(self.rows):
            for col in range(self.cols):
                if self.board[row][col] == 0:
                    continue
                    
                player = self.board[row][col]
                
                for dr, dc in directions:
                    count = 1
                    # Check positive direction
                    r, c = row + dr, col + dc
                    while 0 <= r < self.rows and 0 <= c < self.cols and self.board[r][c] == player:
                        count += 1
                        r, c = r + dr, c + dc
                    
                    # Check negative direction
                    r, c = row - dr, col - dc
                    while 0 <= r < self.rows and 0 <= c < self.cols and self.board[r][c] == player:
                        count += 1
                        r, c = r - dr, c - dc
                    
                    if count >= 4:
                        return player
        
        # Check for draw (board full)
        if all(self.board[0][col] != 0 for col in range(self.cols)):
            return 0
        
        return None  # Game continues
    
    def is_game_over(self) -> bool:
        """Check if the game is over"""
        return self.check_winner() is not None or self._game_over_forced_forfeit
    
    def get_current_player(self) -> int:
        """Get current player (1 for Red, -1 for Yellow)"""
        return self.current_player
    
    def get_state_representation(self) -> str:
        """Get string representation of current game state"""
        board_str = "Current Connect 4 Board State:\n"
        board_str += "  " + " ".join([str(i) for i in range(self.cols)]) + "\n"
        
        for row in range(self.rows):
            row_str = f"{row} "
            for col in range(self.cols):
                symbol = self.get_player_symbol(self.board[row][col])
                row_str += f"{symbol} "
            board_str += row_str + "\n"
        
        current_player_symbol = self.get_player_symbol(self.current_player)
        board_str += f"Current turn: {current_player_symbol} (plays as {self.current_player})\n"
        board_str += f"Legal moves (columns): {self.get_legal_moves()}\n"
        
        return board_str
    
    def get_board_representation_for_llm(self) -> str:
        """Get board state representation for LLM"""
        s = "Current Connect 4 board state:\n"
        s += "Columns: " + " ".join([str(i) for i in range(self.cols)]) + "\n"
        
        for row in range(self.rows):
            row_parts = []
            for col in range(self.cols):
                symbol = self.get_player_symbol(self.board[row][col])
                row_parts.append(f"{symbol}")
            s += "         " + " ".join(row_parts) + "\n"
        
        return s.strip()
    
    def get_board_representation_with_coordinates(self) -> str:
        """Get board state representation with coordinates for each cell"""
        s = "Current Connect 4 board state with coordinates:\n"
        
        for row in range(self.rows):
            row_parts = []
            for col in range(self.cols):
                symbol = self.get_player_symbol(self.board[row][col])
                coord_str = f"({row},{col}): {symbol}"
                row_parts.append(coord_str)
            s += ", ".join(row_parts) + "\n"
        
        return s.strip()
    
    def get_chat_history_for_llm(self, llm: Agent) -> List[dict]:
        """Get prompt for agent describing current game state"""
        board_representation = self.get_board_representation_for_llm()
        player_symbol = self.get_player_symbol(self.current_player)
        legal_moves = self.get_legal_moves()
        legal_moves_str = ", ".join([str(col) for col in legal_moves])
        
        system_prompt = self.system_prompt
        user_prompt_template = self.user_prompt_template
        
        if isinstance(llm, VLLMAgent) and not llm.enable_thinking:
            system_prompt = self.system_prompt_no_thinking
            user_prompt_template = self.user_prompt_template_no_thinking
            
        user_prompt = user_prompt_template.format(
            board_representation=board_representation,
            player_symbol=player_symbol,
            legal_moves_str=legal_moves_str
        )
        
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    
    def parse_move_from_output(self, raw_output: str, legal_moves: List[int]) -> Optional[int]:
        """
        Parse move from LLM output for Connect 4
        
        Args:
            raw_output: Raw text output from the LLM
            legal_moves: List of legal moves for validation
            
        Returns:
            Parsed move as column number or None if no valid move found
        """
        # Look for column numbers in various formats
        # First try to find numbers in brackets
        bracket_match = re.findall(r'\[(\d+)\]', raw_output)
        if bracket_match:
            try:
                move = int(bracket_match[-1])
                if move in legal_moves:
                    return move
                else:
                    logging.warning(f"LLM chose illegal column {move} (not in {legal_moves}). Raw: '{raw_output}'")
                    return None
            except ValueError:
                pass
        
        # Then try to find numbers after thinking block or at end
        think_match = re.search(r'</think>\s*(\d+)', raw_output)
        if think_match:
            try:
                move = int(think_match.group(1))
                if move in legal_moves:
                    return move
                else:
                    logging.warning(f"LLM chose illegal column {move} (not in {legal_moves}). Raw: '{raw_output}'")
                    return None
            except ValueError:
                pass
        
        # Finally, try to find any single digit that's a legal move
        digit_matches = re.findall(r'\b([0-6])\b', raw_output)
        for match in reversed(digit_matches):  # Check from end to beginning
            try:
                move = int(match)
                if move in legal_moves:
                    return move
            except ValueError:
                continue
        
        logging.warning(f"LLM output format incorrect: '{raw_output}'")
        return None
    
    def reset(self) -> None:
        """Reset game to initial state"""
        self.board = [[0 for _ in range(self.cols)] for _ in range(self.rows)]
        self.current_player = 1
        self._game_over_forced_forfeit = False
    
    def force_forfeit(self) -> None:
        """Force game end (for evaluation)"""
        self._game_over_forced_forfeit = True
        
    def clone(self):
        """Create a deep copy of the game"""
        return copy.deepcopy(self)