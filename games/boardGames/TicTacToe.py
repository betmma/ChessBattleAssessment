import copy
import logging
import re
from typing import List, Tuple, Optional, Any
import numpy as np
from games.board_game import BoardGame


class TicTacToeGame(BoardGame, board_size=(3, 3), move_arity=2):
    """Tic-Tac-Toe game implementation as a subclass of BoardGame"""
    
    # Game introduction for the BoardGame system
    game_introduction = (
        "Tic-Tac-Toe is played on a 3x3 grid. "
        "Players take turns placing their symbol (X or O) in empty cells. "
        "The goal is to get three of your symbols in a row, column, or diagonal. "
        "If all cells are filled without a winner, the game is a draw. "
        "Moves are specified as (row, col) coordinates where row and col are 0, 1, or 2."
    )
    
    def __init__(self):
        """Initialize the TicTacToe game"""
        super().__init__()
        self._game_over_forced_forfeit = False
    
    def _get_legal_moves(self) -> List[Tuple[int, int]]:
        """Return all legal moves as (row, col) coordinate pairs"""
        legal_moves = []
        for row in range(3):
            for col in range(3):
                if self.board[row, col] == 0:
                    legal_moves.append((row, col))
        return legal_moves
    
    def make_move(self, move: Tuple[int, int]) -> bool:
        """Execute a move, return True if move was legal and successful"""
        if not isinstance(move, tuple) or len(move) != 2:
            return False
            
        row, col = move
        
        # Validate coordinates
        if not (0 <= row < 3 and 0 <= col < 3):
            return False
            
        # Check if the cell is empty
        if self.board[row, col] != 0:
            return False
            
        # Make the move
        self.board[row, col] = self.current_player
        self.current_player *= -1  # Switch player
        return True
    
    def check_winner(self) -> Optional[int]:
        """Check for a winner, return 1 (X wins), -1 (O wins), 0 (draw), None (game continues)"""
        # Check rows
        for row in range(3):
            if abs(np.sum(self.board[row, :])) == 3:
                return int(np.sum(self.board[row, :]) / 3)
        
        # Check columns
        for col in range(3):
            if abs(np.sum(self.board[:, col])) == 3:
                return int(np.sum(self.board[:, col]) / 3)
        
        # Check diagonals
        diag1_sum = np.sum([self.board[i, i] for i in range(3)])
        if abs(diag1_sum) == 3:
            return int(diag1_sum / 3)
            
        diag2_sum = np.sum([self.board[i, 2-i] for i in range(3)])
        if abs(diag2_sum) == 3:
            return int(diag2_sum / 3)
        
        # Check for draw (board full)
        if not np.any(self.board == 0):
            return 0
        
        # Game continues
        return None
    
    def is_game_over(self) -> bool:
        """Check if the game is over"""
        return self.check_winner() is not None or self._game_over_forced_forfeit
    
    def evaluate_position(self) -> float:
        """
        Evaluate the current position from player 1's perspective.
        Positive values favor player 1 (X), negative values favor player -1 (O).
        """
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
        
        # Lines to check: rows, columns, and diagonals
        lines = []
        
        # Rows
        for row in range(3):
            lines.append([(row, col) for col in range(3)])
        
        # Columns
        for col in range(3):
            lines.append([(row, col) for row in range(3)])
        
        # Diagonals
        lines.append([(i, i) for i in range(3)])  # Main diagonal
        lines.append([(i, 2-i) for i in range(3)])  # Anti-diagonal
        
        for line in lines:
            x_count = sum(1 for pos in line if self.board[pos] == 1)
            o_count = sum(1 for pos in line if self.board[pos] == -1)
            empty_count = sum(1 for pos in line if self.board[pos] == 0)
            
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
    
    def parse_move_from_output(self, raw_output: str) -> Optional[Tuple[int, int]]:
        """
        Parse move from LLM output for Tic-Tac-Toe.
        This overrides the BoardGame method to handle both formats.
        """
        # First try the BoardGame format (inherits tuple parsing)
        move = super().parse_move_from_output(raw_output)
        if move is not None:
            return move
        
        # Also try square bracket format for backward compatibility
        match = re.findall(r'\[\s*([0-2])\s*,\s*([0-2])\s*\]', raw_output)
        if match:
            last_match = match[-1]
            move_coord = (int(last_match[0]), int(last_match[1]))
            return move_coord
        
        logging.warning(f"TicTacToe: Could not parse move from output: '{raw_output}'")
        return None
    
    def reset(self) -> None:
        """Reset game to initial state"""
        super().reset()
        self._game_over_forced_forfeit = False
    
    def force_forfeit(self) -> None:
        """Force game end (for evaluation)"""
        self._game_over_forced_forfeit = True
    
    def clone(self):
        """Create a deep copy of the game"""
        new_game = self.__class__()
        new_game.board = np.copy(self.board)
        new_game.current_player = self.current_player
        new_game._game_over_forced_forfeit = self._game_over_forced_forfeit
        return new_game
