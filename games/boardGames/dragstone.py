import numpy as np
from typing import List, Any, Optional, Tuple
from games.board_game import BoardGame

class DragstoneGame(BoardGame, board_size=(5, 5), move_arity=4):
    """
    Dragstone Game - A unique board game where players drag stones to form patterns.
    
    Rules:
    - 5x5 board with initial stones placed at specific positions
    - Players take turns dragging their stones to adjacent empty cells
    - Goal: Form a line of 3 stones (horizontal, vertical, not including diagonal)
    - Each player starts with 4 stones
    - Stones can only move to adjacent empty cells (8-directional)
    
    Move format: (from_row, from_col, to_row, to_col)
    """
    
    name = "Dragstone"
    game_introduction = (
        "Dragstone is a strategic board game played on a 5x5 grid. "
        "Each player starts with 4 stones positioned on the board. "
        "Players take turns dragging their stones to adjacent empty cells. "
        "The goal is to form a horizontal or vertical line of 3 stones (not including diagonal). "
        "Stones can move to any of the 8 adjacent cells if they are empty. "
        "The game ends when a player forms a line of 3 stones or no legal moves remain. "
        "Move format: (from_row, from_col, to_row, to_col)"
    )
    
    player_symbols = {1: 'X', -1: 'O', 0: '.'}
    
    def __init__(self):
        """Initialize Dragstone game with starting stone positions."""
        super().__init__()
        self._setup_initial_stones()
    
    def _create_initial_board(self) -> np.ndarray:
        """Create empty 5x5 board."""
        return np.zeros((5, 5), dtype=int)
    
    def _setup_initial_stones(self):
        """Set up initial stone positions for both players."""
        # Player 1 (X) starts at top - scattered to avoid immediate win
        self.board[0, 1] = 1
        self.board[0, 3] = 1
        self.board[4, 1] = 1
        self.board[4, 3] = 1
        
        # Player -1 (O) starts at bottom - scattered to avoid immediate win
        self.board[3, 0] = -1
        self.board[3, 4] = -1
        self.board[1, 0] = -1
        self.board[1, 4] = -1
    
    def get_legal_moves(self) -> List[Tuple[int, int, int, int]]:
        """Return all legal drag moves for current player."""
        legal_moves = []
        
        # Find all stones belonging to current player
        player_positions = np.where(self.board == self.current_player)
        
        for i in range(len(player_positions[0])):
            from_row, from_col = player_positions[0][i], player_positions[1][i]
            
            # Check all 8 adjacent positions
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    
                    to_row, to_col = from_row + dr, from_col + dc
                    
                    # Check if destination is within bounds and empty
                    if (0 <= to_row < 5 and 0 <= to_col < 5 and 
                        self.board[to_row, to_col] == 0):
                        legal_moves.append((from_row, from_col, to_row, to_col))
        
        return legal_moves
    
    def make_move(self, move: Tuple[int, int, int, int]) -> bool:
        """Execute a drag move."""
        from_row, from_col, to_row, to_col = move
        
        # Validate move
        if not self._is_valid_move(move):
            return False
        
        # Execute the drag
        self.board[to_row, to_col] = self.current_player
        self.board[from_row, from_col] = 0
        
        # Switch player
        self.current_player = -self.current_player
        
        return True
    
    def _is_valid_move(self, move: Tuple[int, int, int, int]) -> bool:
        """Check if a move is valid."""
        from_row, from_col, to_row, to_col = move
        
        # Check bounds
        if not (0 <= from_row < 5 and 0 <= from_col < 5 and 
                0 <= to_row < 5 and 0 <= to_col < 5):
            return False
        
        # Check if from position has current player's stone
        if self.board[from_row, from_col] != self.current_player:
            return False
        
        # Check if to position is empty
        if self.board[to_row, to_col] != 0:
            return False
        
        # Check if move is to adjacent cell
        dr, dc = abs(to_row - from_row), abs(to_col - from_col)
        if dr > 1 or dc > 1 or (dr == 0 and dc == 0):
            return False
        
        return True
    
    def check_winner(self) -> Optional[int]:
        """Check if there's a winner (3 stones in a line)."""
        # Check all possible lines of 3
        for row in range(5):
            for col in range(5):
                if self.board[row, col] != 0:
                    player = self.board[row, col]
                    
                    # Check horizontal (right)
                    if col <= 2 and self._check_line(row, col, 0, 1, player):
                        return player
                    
                    # Check vertical (down)
                    if row <= 2 and self._check_line(row, col, 1, 0, player):
                        return player
                    
                    # # Check diagonal (down-right)
                    # if row <= 2 and col <= 2 and self._check_line(row, col, 1, 1, player):
                    #     return player
                    
                    # # Check diagonal (down-left)
                    # if row <= 2 and col >= 2 and self._check_line(row, col, 1, -1, player):
                    #     return player
        
        return None
    
    def _check_line(self, start_row: int, start_col: int, dr: int, dc: int, player: int) -> bool:
        """Check if 3 stones of the same player form a line."""
        for i in range(3):
            r, c = start_row + i * dr, start_col + i * dc
            if self.board[r, c] != player:
                return False
        return True
    
    def is_game_over(self) -> bool:
        """Check if the game is over."""
        # Check for winner
        if self.check_winner() is not None:
            return True
        
        # Check if current player has any legal moves
        return len(self.get_legal_moves()) == 0
    
    def evaluate_position(self) -> float:
        """Evaluate the current position for minimax."""
        winner = self.check_winner()
        if winner == 1:
            return 1000.0
        elif winner == -1:
            return -1000.0
        
        score = 0.0
        
        # Count potential lines for each player
        for player in [1, -1]:
            player_score = 0
            
            # Check all possible lines of 3
            for row in range(5):
                for col in range(5):
                    # Horizontal lines
                    if col <= 2:
                        line_score = self._evaluate_line(row, col, 0, 1, player)
                        player_score += line_score
                    
                    # Vertical lines
                    if row <= 2:
                        line_score = self._evaluate_line(row, col, 1, 0, player)
                        player_score += line_score
                    
                    # # Diagonal lines
                    # if row <= 2 and col <= 2:
                    #     line_score = self._evaluate_line(row, col, 1, 1, player)
                    #     player_score += line_score
                    
                    # if row <= 2 and col >= 2:
                    #     line_score = self._evaluate_line(row, col, 1, -1, player)
                    #     player_score += line_score
            
            if player == 1:
                score += player_score
            else:
                score -= player_score
        
        # Add mobility bonus (more moves = better position)
        current_moves = len(self.get_legal_moves())
        self.current_player = -self.current_player
        opponent_moves = len(self.get_legal_moves())
        self.current_player = -self.current_player
        
        if self.current_player == 1:
            score += 0.1 * (current_moves - opponent_moves)
        else:
            score -= 0.1 * (current_moves - opponent_moves)
        
        return score
    
    def _evaluate_line(self, start_row: int, start_col: int, dr: int, dc: int, player: int) -> float:
        """Evaluate a potential line of 3 for a player."""
        stones = 0
        empty = 0
        blocked = False
        
        for i in range(3):
            r, c = start_row + i * dr, start_col + i * dc
            cell = self.board[r, c]
            
            if cell == player:
                stones += 1
            elif cell == 0:
                empty += 1
            else:
                blocked = True
                break
        
        if blocked:
            return 0.0
        
        if stones == 3:
            return 100.0  # Winning line
        elif stones == 2 and empty == 1:
            return 10.0   # Two in a row with one empty
        elif stones == 1 and empty == 2:
            return 1.0    # One stone with two empty
        else:
            return 0.0
    
    def reset(self) -> None:
        """Reset the game to initial state."""
        super().reset()
        self._setup_initial_stones()
    
    def parse_move_from_string(self, move_str: str) -> Optional[Tuple[int, int, int, int]]:
        """Parse a move string back to a tuple format."""
        try:
            # Remove parentheses and split by comma
            move_str = move_str.strip('()')
            parts = [int(x.strip()) for x in move_str.split(',')]
            if len(parts) == 4:
                return tuple(parts)
        except:
            pass
        return None
