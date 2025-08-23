import numpy as np
from typing import List, Any, Optional, Tuple
from games.board_game import BoardGame

class Mut153Game(BoardGame, board_size=(3, 3), move_arity=2):
    name = "Mutation 153"
    game_introduction = """
    Mutation 153 is a 2-player abstract strategy game played on a 3x3 board. Each cell starts with a value of 3. On your turn, you select a cell to mutate. To mutate a cell, all adjacent cells must have at least 1. When you mutate a cell, its value increases by 1, and each adjacent cell's value decreases by 1. The game ends when no legal moves remain. The last player to make a move wins.
    """
    player_symbols = {1: 'X', -1: 'O', 0: '.'}  # Default symbols, not used in representation

    def _create_initial_board(self) -> np.ndarray:
        return np.full(self.board_size, 3, dtype=int)

    def _get_legal_moves(self) -> List[Tuple[int, int]]:
        legal_moves = []
        rows, cols = self.board_size
        for r in range(rows):
            for c in range(cols):
                adj_cells = []
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        adj_cells.append(self.board[nr, nc])
                if all(cell >= 1 for cell in adj_cells):
                    legal_moves.append((r, c))
        return legal_moves

    def make_move(self, move: Tuple[int, int]) -> bool:
        r, c = move
        rows, cols = self.board_size
        adj_cells = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                adj_cells.append((nr, nc))
        self.board[r, c] += 1
        for nr, nc in adj_cells:
            self.board[nr, nc] -= 1
        self.current_player *= -1
        return True

    def check_winner(self) -> Optional[Any]:
        if not self._get_legal_moves():
            return -self.current_player
        return None

    def is_game_over(self) -> bool:
        return len(self._get_legal_moves()) == 0

    def evaluate_position(self) -> float:
        total_sum = self.board.sum()
        return total_sum * self.current_player

    def get_board_representation_for_llm(self) -> str:
        return "\n".join([" ".join(map(str, row)) for row in self.board])