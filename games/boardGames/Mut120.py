
import numpy as np
from typing import List, Any, Optional, Tuple
from games.board_game import BoardGame

class Mut120(BoardGame, board_size=(3, 3), move_arity=2):
    name = "Mut120"
    game_introduction = """
Mut120 is a strategic abstract game played on a 3x3 grid. Each cell starts with 3 tokens. Players take turns removing all tokens from a single cell. When a cell is emptied, each adjacent cell (up, down, left, right) loses one token. The game ends when all cells are empty. The last player to make a move wins. Moves are specified as (row, column) tuples, with rows and columns numbered from 0 to 2.
"""

    def _create_initial_board(self) -> np.ndarray:
        return np.full(self.board_size, 3, dtype=int)

    def _get_legal_moves(self) -> List[Tuple[int, int]]:
        legal_moves = []
        for row in range(self.board.shape[0]):
            for col in range(self.board.shape[1]):
                if self.board[row, col] > 0:
                    legal_moves.append((row, col))
        return legal_moves

    def make_move(self, move: Tuple[int, int]) -> bool:
        row, col = move
        if self.board[row, col] <= 0:
            return False
        # Remove all tokens from the selected cell
        self.board[row, col] = 0
        # Apply the effect to adjacent cells
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dr, dc in directions:
            nr, nc = row + dr, col + dc
            if 0 <= nr < self.board.shape[0] and 0 <= nc < self.board.shape[1]:
                self.board[nr, nc] = max(0, self.board[nr, nc] - 1)
        # Switch player
        self.current_player *= -1
        return True

    def check_winner(self) -> Optional[Any]:
        if np.all(self.board == 0):
            # The last player to make a move wins
            return self.current_player * -1
        return None

    def is_game_over(self) -> bool:
        return np.all(self.board == 0)

    def evaluate_position(self) -> float:
        # Heuristic: Check if current player can force a win in one move
        for move in self._get_legal_moves():
            new_game = self.clone()
            new_game.make_move(move)
            if new_game.is_game_over():
                winner = new_game.check_winner()
                if winner == self.current_player:
                    return 1.0
                elif winner == -self.current_player:
                    return -1.0
        return 0.0
