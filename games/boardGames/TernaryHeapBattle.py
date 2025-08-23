
import numpy as np
from typing import List, Any, Optional, Tuple
from games.board_game import BoardGame

class TernaryHeapBattle(BoardGame, board_size=(3,), move_arity=2):
    name = "Ternary Heap Battle"
    game_introduction = (
        "Ternary Heap Battle is a mathematical game played with three heaps of stones. "
        "Each heap starts with 3 stones. Players take turns to remove either 1 or 2 stones from a single heap. "
        "The player who removes the last stone wins. The game ends when all heaps are empty. "
        "Moves are represented as tuples (heap_index, stones_to_remove), where heap_index is 0-2 and stones_to_remove is 1 or 2."
    )

    def _create_initial_board(self) -> np.ndarray:
        return np.array([3, 3, 3], dtype=int)

    def _get_legal_moves(self) -> List[Any]:
        legal_moves = []
        for heap_idx in range(3):
            stones = self.board[heap_idx]
            if stones >= 1:
                legal_moves.append((heap_idx, 1))
            if stones >= 2:
                legal_moves.append((heap_idx, 2))
        return legal_moves

    def make_move(self, move: Any) -> bool:
        heap_idx, stones = move
        if self.board[heap_idx] >= stones:
            self.board[heap_idx] -= stones
            self.current_player *= -1
            return True
        return False

    def check_winner(self) -> Optional[Any]:
        if np.all(self.board == 0):
            return self.current_player * -1  # Last player to move wins
        return None

    def is_game_over(self) -> bool:
        return np.all(self.board == 0)

    def evaluate_position(self) -> float:
        # Simple evaluation: difference in total stones between players
        return float(np.sum(self.board) * self.current_player)
