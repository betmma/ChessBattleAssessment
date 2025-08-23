import numpy as np
from typing import List, Any, Optional, Tuple
from games.board_game import BoardGame

class TriadTakeaway(BoardGame, board_size=(3, 3), move_arity=3):
    game_introduction = """
Triad Takeaway is a 2-player abstract strategy game played on a 3x3 grid. Each cell starts with 2 stones. On your turn, select a cell (i,j) and choose to remove 1 or 2 stones from it. After doing so, remove 1 stone from each adjacent cell (up, down, left, right) if they contain at least 1 stone. The player who removes the last stone wins.
    """
    name = "Triad Takeaway"

    def _create_initial_board(self) -> np.ndarray:
        return np.full(self.board_size, 2, dtype=int)

    def _get_legal_moves(self) -> List[Tuple[int, int, int]]:
        moves = []
        for i in range(3):
            for j in range(3):
                stones = self.board[i, j]
                if stones >= 1:
                    moves.append((i, j, 1))
                if stones >= 2:
                    moves.append((i, j, 2))
        return moves

    def make_move(self, move: Tuple[int, int, int]) -> bool:
        i, j, k = move
        if self.board[i, j] < k:
            return False
        # Remove k stones from selected cell
        self.board[i, j] -= k
        # Remove 1 stone from adjacent cells
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dx, dy in directions:
            x, y = i + dx, j + dy
            if 0 <= x < 3 and 0 <= y < 3 and self.board[x, y] >= 1:
                self.board[x, y] -= 1
        # Switch player
        self.current_player *= -1
        return True

    def check_winner(self) -> Optional[int]:
        if np.all(self.board == 0):
            # Last move emptied the board, previous player wins
            return self.current_player * -1
        return None

    def is_game_over(self) -> bool:
        return np.all(self.board == 0) or not self._get_legal_moves()