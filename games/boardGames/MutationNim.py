import numpy as np
from typing import List, Any, Optional, Tuple
from games.board_game import BoardGame
import numpy as np
from typing import List, Any, Optional, Tuple
from games.board_game import BoardGame

class MutationNim(BoardGame, board_size=(3,3), move_arity=2):
    name = "Mutation Nim"
    game_introduction = (
        "Mutation Nim is a strategic game played on a 3x3 grid. Each cell starts with 3 tokens. "
        "Players take turns selecting a cell to remove 1 token from it and 1 token from each adjacent cell (up, down, left, right). "
        "A cell cannot have fewer than 0 tokens. The player who makes the last move wins the game. "
        "Moves are in the format: (row, column), where rows and columns are 0-based indices."
    )

    def _create_initial_board(self) -> np.ndarray:
        return np.full(self.board_size, 3, dtype=int)

    def _get_legal_moves(self) -> List[Tuple[int, int]]:
        legal_moves = []
        for i in range(self.board.shape[0]):
            for j in range(self.board.shape[1]):
                if self.board[i][j] >= 1:
                    legal_moves.append((i, j))
        return legal_moves

    def make_move(self, move: Tuple[int, int]) -> bool:
        row, col = move
        if self.board[row][col] == 0:
            return False
        # Reduce the selected cell
        self.board[row][col] -= 1
        self.board[row][col] = max(0, self.board[row][col])
        # Reduce adjacent cells
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dr, dc in directions:
            nr, nc = row + dr, col + dc
            if 0 <= nr < 3 and 0 <= nc < 3:
                self.board[nr][nc] -= 1
                self.board[nr][nc] = max(0, self.board[nr][nc])
        # Switch player
        self.current_player *= -1
        return True

    def check_winner(self) -> Optional[Any]:
        if np.all(self.board == 0):
            # The last player to make a move is the winner
            return -self.current_player
        return None

    def is_game_over(self) -> bool:
        return np.all(self.board == 0)

    def evaluate_position(self) -> float:
        # Evaluation: sum of tokens * current player
        total_tokens = np.sum(self.board)
        return float(total_tokens * self.current_player)