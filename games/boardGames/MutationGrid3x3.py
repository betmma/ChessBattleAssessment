
import numpy as np
from typing import List, Any, Optional, Tuple
from games.board_game import BoardGame

class MutationGrid3x3(BoardGame, board_size=(3, 3), move_arity=2):
    name = "MutationGrid3x3"
    game_introduction = """
Mutation Grid 3x3 is an abstract strategy game played on a 3x3 grid. Each cell starts with the value 1. On your turn, select a cell (row, column). The selected cell's value decreases by 1, while each of its adjacent cells (up, down, left, right) increases by 1. A move is only legal if, after applying it, no cell exceeds the maximum value of 3. The game ends when a player cannot make a legal move. The last player to successfully make a move wins the game.
"""
    player_symbols = {1: 'P1', -1: 'P2'}  # Not used in representation; overridden in get_board_representation_for_llm

    def get_board_representation_for_llm(self) -> str:
        if self.board.ndim == 1:
            return " ".join(str(cell) for cell in self.board)
        else:
            return "\n".join(" ".join(str(cell) for cell in row) for row in self.board)

    def _create_initial_board(self) -> np.ndarray:
        return np.ones(self.board_size, dtype=int)

    def _get_legal_moves(self) -> List[Tuple[int, int]]:
        legal_moves = []
        max_value = 3
        rows, cols = self.board_size
        for i in range(rows):
            for j in range(cols):
                temp_board = self.board.copy()
                temp_board[i, j] -= 1
                for di, dj in [(-1,0), (1,0), (0,-1), (0,1)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < rows and 0 <= nj < cols:
                        temp_board[ni, nj] += 1
                if np.all(temp_board <= max_value):
                    legal_moves.append((i, j))
        return legal_moves

    def make_move(self, move: Tuple[int, int]) -> bool:
        i, j = move
        rows, cols = self.board_size
        if not (0 <= i < rows and 0 <= j < cols):
            return False
        self.board[i, j] -= 1
        for di, dj in [(-1,0), (1,0), (0,-1), (0,1)]:
            ni, nj = i + di, j + dj
            if 0 <= ni < rows and 0 <= nj < cols:
                self.board[ni, nj] += 1
        self.current_player *= -1
        return True

    def check_winner(self) -> Optional[Any]:
        if not self.is_game_over():
            return None
        return -self.current_player

    def is_game_over(self) -> bool:
        return len(self._get_legal_moves()) == 0
