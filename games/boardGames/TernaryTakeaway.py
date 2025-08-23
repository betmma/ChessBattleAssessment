import numpy as np
from typing import List, Any, Optional, Tuple
from games.board_game import BoardGame

class TernaryTakeaway(BoardGame, board_size=(3, 3), move_arity=2):
    game_introduction = """
Ternary Takeaway is a two-player abstract strategy game played on a 3x3 grid. Each cell contains a token count of 0, 1, or 2. Players take turns removing tokens from a cell. The game ends when all cells are empty, and the last player to make a move wins.

Rules:
1. The board is a 3x3 grid initialized with all cells set to 2.
2. On a turn, a player selects a cell with at least 1 token.
3. If the selected cell has 2 tokens:
   a. Remove one token, reducing it to 1.
   b. Remove one token from each of the four adjacent cells (up, down, left, right), if they exist and have at least 1 token.
4. If the selected cell has 1 token:
   a. Remove it, reducing it to 0.
5. The game ends when all cells are 0.
6. The player who makes the last move (i.e., the one who causes all cells to be 0) wins.

Move Format:
Each move is a tuple of two integers (row, column), where rows and columns are 0-indexed.
"""

    def __init__(self):
        super().__init__()
        self.winner = None

    def _create_initial_board(self) -> np.ndarray:
        return np.full(self.board_size, 2, dtype=int)

    def _get_legal_moves(self) -> List[Any]:
        moves = []
        for i in range(3):
            for j in range(3):
                if self.board[i, j] > 0:
                    moves.append((i, j))
        return moves

    def make_move(self, move: Any) -> bool:
        row, col = move
        if not (0 <= row < 3 and 0 <= col < 3):
            return False
        if self.board[row, col] == 0:
            return False
        original_value = self.board[row, col]
        if original_value == 2:
            self.board[row, col] = 1
            # Reduce adjacent cells
            for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                nr, nc = row + dr, col + dc
                if 0 <= nr < 3 and 0 <= nc < 3:
                    self.board[nr, nc] = max(0, self.board[nr, nc] - 1)
        elif original_value == 1:
            self.board[row, col] = 0
        else:
            return False  # should not happen if move is legal
        # Check if game is over
        if self.is_game_over():
            self.winner = self.current_player
        else:
            self.current_player *= -1
        return True

    def is_game_over(self) -> bool:
        return np.all(self.board == 0)

    def check_winner(self) -> Optional[Any]:
        if self.is_game_over():
            return self.winner
        return None

    def evaluate_position(self) -> float:
        # Positive for current player advantage
        total_tokens = np.sum(self.board)
        return total_tokens * self.current_player

    def get_board_representation_for_llm(self) -> str:
        symbols = {1: '1', 2: '2', 0: '.'}
        return "\n".join([
            " ".join([symbols[cell] for cell in row])
            for row in self.board
        ])