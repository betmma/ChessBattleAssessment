
import numpy as np
from typing import List, Any, Optional, Tuple
from games.board_game import BoardGame

class AdjacencyNim(BoardGame, board_size=(3, 3), move_arity=3):
    game_introduction = (
        "Adjacency Nim is a two-player abstract strategy game played on a 3x3 grid. "
        "Each cell starts with 3 stones. On your turn, select a cell and remove 1 to K stones, "
        "where K is the number of adjacent cells for that cell (corners have 2, edges have 3, center has 4). "
        "You must remove at least 1 stone and at most K stones, but not exceeding the current number in the cell. "
        "The player who removes the last stone wins the game. "
        "A move is specified as a tuple (i, j, s), where (i,j) are the cell coordinates and s is the number of stones to remove."
    )

    def _get_legal_moves(self) -> List[Any]:
        legal_moves = []
        for i in range(3):
            for j in range(3):
                v = self.board[i, j]
                if v > 0:
                    # Determine the number of adjacent cells (k)
                    if (i == 0 or i == 2) and (j == 0 or j == 2):
                        # corner
                        k = 2
                    elif i == 1 and j == 1:
                        # center
                        k = 4
                    else:
                        # edge
                        k = 3
                    max_stones = min(v, k)
                    for s in range(1, max_stones + 1):
                        legal_moves.append((i, j, s))
        return legal_moves

    def make_move(self, move: Any) -> bool:
        i, j, s = move
        if self.board[i, j] < s:
            return False
        self.board[i, j] -= s
        self.current_player *= -1
        return True

    def check_winner(self) -> Optional[Any]:
        if np.all(self.board == 0):
            return self.current_player * -1  # The last player to move wins
        return None

    def is_game_over(self) -> bool:
        return np.all(self.board == 0)

    def _create_initial_board(self) -> np.ndarray:
        return np.full(self.board_size, 3, dtype=int)

    def evaluate_position(self) -> float:
        # Simple heuristic: difference in total stones between players
        # Assuming that the player with more moves available has an advantage
        return (np.sum(self.board) / 2) * self.current_player
