import numpy as np
from typing import List, Any, Optional, Tuple
from games.board_game import BoardGame

class MutationGridGame(BoardGame, board_size=(3,3), move_arity=2):
    game_introduction = (
        "Mutation Grid is a 3x3 grid game where each cell starts with a value "
        "(edges 1, center 3). On your turn, select a cell to decrease its value "
        "by 1 and increase each adjacent cell by 1. However, no cell can exceed "
        "a maximum value of 5. The game ends when no player can move. "
        "The last player to have moved wins. Moves are 2-tuples (row, column) "
        "with 0-based indices."
    )
    max_cell_value = 5

    def _create_initial_board(self) -> np.ndarray:
        return np.array([[1,1,1],
                         [1,3,1],
                         [1,1,1]], dtype=int)

    def _get_legal_moves(self) -> List[Any]:
        legal_moves = []
        rows, cols = self.board_size
        for i in range(rows):
            for j in range(cols):
                if self.board[i, j] == 0:
                    continue
                # Check if selecting this cell is possible
                temp_board = self.board.copy()
                temp_board[i, j] -= 1
                if temp_board[i, j] < 0:
                    continue
                # Check adjacent cells
                directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
                valid = True
                for dx, dy in directions:
                    x, y = i + dx, j + dy
                    if 0 <= x < rows and 0 <= y < cols:
                        if temp_board[x, y] + 1 > self.max_cell_value:
                            valid = False
                            break
                if valid:
                    legal_moves.append((i, j))
        return legal_moves

    def make_move(self, move: Any) -> bool:
        i, j = move
        # Apply the move
        self.board[i, j] -= 1
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dx, dy in directions:
            x, y = i + dx, j + dy
            if 0 <= x < 3 and 0 <= y < 3:
                self.board[x, y] += 1
        # Switch player
        self.current_player *= -1
        return True

    def check_winner(self) -> Optional[Any]:
        if not self.is_game_over():
            return None
        # If game is over, the current player cannot move, so the other player wins
        return -self.current_player

    def is_game_over(self) -> bool:
        return len(self._get_legal_moves()) == 0