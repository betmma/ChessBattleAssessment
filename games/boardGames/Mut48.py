
import numpy as np
from typing import List, Any, Optional, Tuple
from games.board_game import BoardGame

class Mut48(BoardGame, board_size=(4, 4), move_arity=2):
    name = "Mut48"
    game_introduction = """
    Mut48 is a strategic abstract game played on a 4x4 grid. Each cell starts with 3 tokens (total 48). 
    Players take turns selecting a cell with at least 1 token. On your turn:
    1. Remove 1 token from the selected cell.
    2. If the remaining tokens in that cell are even, also remove 1 token from each adjacent cell (up, down, left, right).
    The game ends when no cell has at least 1 token. The last player to make a move wins.
    Move format: (row, column), where row and column are 0-based integers.
    """
    player_symbols = {1: 'A', -1: 'B', 0: '.'}  # Example symbols for players; actual board shows token counts as numbers

    def _create_initial_board(self) -> np.ndarray:
        return np.full(self.board_size, 3, dtype=int)

    def _get_legal_moves(self) -> List[Tuple[int, int]]:
        moves = []
        for row in range(self.board.shape[0]):
            for col in range(self.board.shape[1]):
                if self.board[row, col] >= 1:
                    moves.append((row, col))
        return moves

    def make_move(self, move: Tuple[int, int]) -> bool:
        row, col = move
        if self.board[row, col] < 1:
            return False
        self.board[row, col] -= 1
        if self.board[row, col] % 2 == 0:
            directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            for dr, dc in directions:
                r, c = row + dr, col + dc
                if 0 <= r < self.board.shape[0] and 0 <= c < self.board.shape[1]:
                    self.board[r, c] -= 1
        self.current_player *= -1
        return True

    def check_winner(self) -> Optional[Any]:
        if not self.is_game_over():
            return None
        return self.current_player * -1  # Last player to move wins

    def is_game_over(self) -> bool:
        return np.all(self.board <= 0)

    def evaluate_position(self) -> float:
        if self.is_game_over():
            return float('inf') if self.check_winner() == 1 else float('-inf')
        return 0.0
