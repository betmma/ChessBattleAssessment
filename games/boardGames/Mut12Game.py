
import numpy as np
from typing import List, Any, Optional, Tuple
from games.board_game import BoardGame

class Mut12Game(BoardGame, board_size=(4,), move_arity=2):
    name = "Mut_12"
    game_introduction = """
Mut_12 is an abstract strategy game for two players. The game is played on a 1D board of four cells, each containing a non-negative integer. Players take turns selecting a cell and performing one of two actions:

1. Subtract 1 from the selected cell and add 1 to the next cell (if it's not the last cell).
2. Subtract 2 from the selected cell.

A move is legal if the selected cell's value is sufficient for the action (at least 1 for action 0, at least 2 for action 1). The game ends when all cells are zero. The player who makes the last move wins.

Moves are specified as a tuple (cell_index, action), where cell_index is 0-3 and action is 0 (subtract and pass) or 1 (subtract 2).
"""

    def _create_initial_board(self) -> np.ndarray:
        return np.array([3, 3, 3, 3], dtype=int)

    def _get_legal_moves(self) -> List[Tuple[int, int]]:
        legal_moves = []
        for i in range(4):
            val = self.board[i]
            if val >= 1:
                legal_moves.append((i, 0))
            if val >= 2:
                legal_moves.append((i, 1))
        return legal_moves

    def make_move(self, move: Tuple[int, int]) -> bool:
        i, action = move
        if i < 0 or i >= 4 or action not in (0, 1):
            return False
        val = self.board[i]
        if (action == 0 and val < 1) or (action == 1 and val < 2):
            return False

        # Apply the move
        if action == 0:
            self.board[i] -= 1
            if i < 3:  # Not the last cell
                self.board[i + 1] += 1
        else:  # action == 1
            self.board[i] -= 2

        # Check if game is over
        if np.all(self.board == 0):
            return True  # Game over, no player switch

        # Switch players
        self.current_player *= -1
        return True

    def is_game_over(self) -> bool:
        return np.all(self.board == 0)

    def check_winner(self) -> Optional[Any]:
        if self.is_game_over():
            return self.current_player
        return None

    def evaluate_position(self) -> float:
        # Simple evaluation based on parity of total tokens
        total_tokens = np.sum(self.board)
        return float(total_tokens % 2) * 2 - 1  # Favor player with advantage
