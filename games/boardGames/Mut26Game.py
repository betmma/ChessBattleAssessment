
import numpy as np
from typing import List, Any, Optional, Tuple
from games.board_game import BoardGame

class Mut26Game(BoardGame, board_size=(4, 4), move_arity=2):
    game_introduction = """
    Mut26 is a 2-player abstract strategy game played on a 4x4 grid. Each cell starts with a value of 2. On your turn, you select a cell (x, y) to mutate. When you mutate a cell, its value increases by 1, and each of its adjacent cells (up, down, left, right) decreases by 1. A move is only legal if all adjacent cells have a value of at least 1 before the move (so they can be decreased to 0). The game ends when no player can make a move. The last player to successfully make a move wins the game.
    """

    def _create_initial_board(self) -> np.ndarray:
        return np.full(self.board_size, 2, dtype=int)

    def _get_legal_moves(self) -> List[Tuple[int, int]]:
        legal_moves = []
        rows, cols = self.board_size
        for x in range(rows):
            for y in range(cols):
                is_legal = True
                # Check up
                if x > 0 and self.board[x-1, y] < 1:
                    is_legal = False
                # Check down
                if x < rows-1 and self.board[x+1, y] < 1:
                    is_legal = False
                # Check left
                if y > 0 and self.board[x, y-1] < 1:
                    is_legal = False
                # Check right
                if y < cols-1 and self.board[x, y+1] < 1:
                    is_legal = False
                if is_legal:
                    legal_moves.append((x, y))
        return legal_moves

    def make_move(self, move: Tuple[int, int]) -> bool:
        x, y = move
        self.board[x, y] += 1
        # Decrease adjacent cells
        if x > 0:
            self.board[x-1, y] -= 1
        if x < self.board.shape[0] - 1:
            self.board[x+1, y] -= 1
        if y > 0:
            self.board[x, y-1] -= 1
        if y < self.board.shape[1] - 1:
            self.board[x, y+1] -= 1
        self.current_player *= -1
        return True

    def check_winner(self) -> Optional[Any]:
        if not self.is_game_over():
            return None
        # Last player to move wins
        return -self.current_player

    def is_game_over(self) -> bool:
        return len(self._get_legal_moves()) == 0
