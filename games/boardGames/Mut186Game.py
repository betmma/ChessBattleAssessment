
import numpy as np
from typing import List, Any, Optional, Tuple
from games.board_game import BoardGame

class Mut186Game(BoardGame):
    name = "Mut-186"
    board_size = (3, 3)
    move_arity = 2
    game_introduction = (
        "Mut-186 is a strategic token manipulation game played on a 3x3 grid. "
        "Each cell starts with 2 tokens. On your turn, select a cell with at least one token. "
        "Remove all tokens from that cell and distribute them equally to its adjacent cells. "
        "If the number of tokens is not divisible by the number of adjacent cells, the remainder is lost. "
        "The game ends when no cell has tokens left. The last player to make a move wins. "
        "Moves are in the format (x, y), where x and y are 0-based indices."
    )

    def _create_initial_board(self) -> np.ndarray:
        return np.full(self.board_size, 2, dtype=int)

    def _get_legal_moves(self) -> List[Any]:
        moves = []
        for x in range(self.board.shape[0]):
            for y in range(self.board.shape[1]):
                if self.board[x, y] > 0:
                    moves.append((x, y))
        return moves

    def make_move(self, move: Any) -> bool:
        x, y = move
        if self.board[x, y] <= 0:
            return False
        value = self.board[x, y]
        self.board[x, y] = 0
        adj_cells = self._get_adjacent_cells(x, y)
        tokens_per_adj = value // len(adj_cells)
        for adj_x, adj_y in adj_cells:
            self.board[adj_x, adj_y] += tokens_per_adj
        self.current_player *= -1
        return True

    def _get_adjacent_cells(self, x: int, y: int) -> List[Tuple[int, int]]:
        adj = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.board.shape[0] and 0 <= ny < self.board.shape[1]:
                adj.append((nx, ny))
        return adj

    def is_game_over(self) -> bool:
        return np.all(self.board == 0)

    def check_winner(self) -> Optional[Any]:
        if self.is_game_over():
            return self.current_player * -1  # The last player to make a move wins
        return None

    def evaluate_position(self) -> float:
        legal_moves = self.get_legal_moves()
        return len(legal_moves) if self.current_player == 1 else -len(legal_moves)
