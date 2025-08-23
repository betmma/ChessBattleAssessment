
import numpy as np
from typing import List, Any, Optional, Tuple
from games.board_game import BoardGame

class Mut132Game(BoardGame, board_size=(3,), move_arity=2):
    name = "Mut132"
    game_introduction = (
        "Mut132 is a three-pile game where players take turns removing 1, 2, or 3 stones from a single pile. "
        "The player who removes the last stone(s) wins. "
        "On your turn, choose a pile (0, 1, or 2) and specify the number of stones (1, 2, or 3) to remove. "
        "Moves are in the format: (pile_index, amount)."
    )

    def _create_initial_board(self) -> np.ndarray:
        return np.array([13, 13, 13], dtype=int)

    def _get_legal_moves(self) -> List[Tuple[int, int]]:
        moves = []
        for pile_index in range(3):
            pile_value = self.board[pile_index]
            for amount in [1, 2, 3]:
                if pile_value >= amount:
                    moves.append((pile_index, amount))
        return moves

    def make_move(self, move: Tuple[int, int]) -> bool:
        pile_index, amount = move
        if pile_index < 0 or pile_index >= 3:
            return False
        if amount not in [1, 2, 3]:
            return False
        if self.board[pile_index] < amount:
            return False
        self.board[pile_index] -= amount
        self.current_player *= -1
        return True

    def check_winner(self) -> Optional[Any]:
        if np.all(self.board == 0):
            # Last player to move wins (current_player has already been switched)
            return -self.current_player
        return None

    def is_game_over(self) -> bool:
        return np.all(self.board == 0)

    def evaluate_position(self) -> float:
        # Heuristic: Sum of remaining stones weighted by current player
        return float(np.sum(self.board)) * self.current_player
