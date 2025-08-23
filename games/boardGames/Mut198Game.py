import numpy as np
from typing import List, Any, Optional, Tuple
from games.board_game import BoardGame

class Mut198Game(BoardGame, board_size=(5,), move_arity=2):
    name = "Mut198"
    game_introduction = (
        "Mut198 is a strategic number mutation game for two players. The board consists of five numbered cells "
        "each containing a value from 0 to 4. On your turn, you select a cell and either: "
        "(1) Decrease its value by 1 (minimum 0), or (2) Split its value into two adjacent cells, "
        "transferring half the value (rounded down) to the left and right neighbors (if they exist). "
        "The game ends when all cells are 0, and the last player to make a move wins. "
        "Move format: (cell_index, action), where action is 0 for decrement, or 1 for split."
    )

    def _create_initial_board(self) -> np.ndarray:
        return np.array([2, 2, 2, 2, 2], dtype=int)

    def _get_legal_moves(self) -> List[Any]:
        moves = []
        for i in range(5):
            val = self.board[i]
            # Decrement action (0) if value > 0
            if val > 0:
                moves.append((i, 0))
            # Split action (1) if value >= 2 and has space to split
            if val >= 2:
                moves.append((i, 1))
        return moves

    def make_move(self, move: Any) -> bool:
        i, action = move
        if action == 0:  # Decrement
            if self.board[i] <= 0:
                return False
            self.board[i] -= 1
        else:  # Split
            if self.board[i] < 2:
                return False
            val = self.board[i]
            half = val // 2
            # Distribute to left
            if i > 0:
                self.board[i-1] += half
            # Distribute to right
            if i < 4:
                self.board[i+1] += val - half
            self.board[i] = 0
        # Switch player
        self.current_player *= -1
        return True

    def check_winner(self) -> Optional[Any]:
        if np.all(self.board == 0):
            # Last player to move wins
            return -self.current_player
        return None

    def is_game_over(self) -> bool:
        return np.all(self.board == 0) or not self.get_legal_moves()

    def evaluate_position(self) -> float:
        # Simple evaluation: difference in potential moves
        return float(np.sum(self.board)) * self.current_player