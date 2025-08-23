
import numpy as np
from typing import List, Any, Optional, Tuple
from games.board_game import BoardGame

class TriNim(BoardGame, board_size=(6,), move_arity=2):
    """
    TriNim is a triangular Nim game played on a 1D board of 6 cells representing rows 0-5.
    Each row i contains i+1 stones. Players take turns removing 1-3 stones from a single row.
    The player who takes the last stone wins.
    """
    name = "TriNim"
    game_introduction = """
    TriNim is a strategic stone-removal game played on a triangular grid. The board consists of 6 rows:
    Row 0 has 1 stone, row 1 has 2 stones, ..., row 5 has 6 stones. Players take turns removing 1-3 stones from a single row.
    The player who removes the last stone wins.

    Move Format: (row_index, stones_to_remove), where row_index is 0-5 and stones_to_remove is 1-3.
    """

    def _create_initial_board(self) -> np.ndarray:
        """Initialize the board with stones in each row (row i has i+1 stones)."""
        return np.array([i+1 for i in range(6)], dtype=int)

    def _get_legal_moves(self) -> List[Tuple[int, int]]:
        """Return all legal (row, count) moves where count is 1-3 and <= current stones in the row."""
        legal_moves = []
        for row in range(6):
            stones = self.board[row]
            if stones == 0:
                continue
            max_take = min(3, stones)
            for take in range(1, max_take + 1):
                legal_moves.append((row, take))
        return legal_moves

    def make_move(self, move: Tuple[int, int]) -> bool:
        """Remove stones from the specified row. Return True if successful."""
        row, take = move
        if not (0 <= row < 6) or not (1 <= take <= 3):
            return False
        if self.board[row] < take:
            return False
        self.board[row] -= take
        self.current_player *= -1
        return True

    def check_winner(self) -> Optional[Any]:
        """Return the winner (1 or -1) if the game is over, else None."""
        if self.is_game_over():
            # The player who made the last move (opposite of current_player) wins
            return -self.current_player
        return None

    def is_game_over(self) -> bool:
        """Check if all rows are empty."""
        return np.all(self.board == 0)
