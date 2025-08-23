
import numpy as np
from typing import List, Any, Optional, Tuple
from games.board_game import BoardGame

class TriadMutation(BoardGame, board_size=(3, 3), move_arity=2):
    """
    Triad Mutation is a 3x3 grid game for two players. 
    Players take turns placing their symbol (X or O) on an empty cell.
    When a player places their symbol in a cell, all adjacent cells 
    (horizontally and vertically) are flipped:
    - Adjacent cells with the same symbol become empty.
    - Adjacent cells with the opposite symbol become the player's symbol.
    - Empty cells remain unchanged.
    The game ends when no empty cells remain or when a player cannot make a move. 
    The player who makes the last move wins.
    """
    game_introduction = """
    Triad Mutation is a strategic 3x3 grid game for two players. Players take turns placing their symbol (X or O) on an empty cell. When a player places their symbol, all adjacent cells (horizontally and vertically) are flipped according to these rules:
    - Adjacent cells with the same symbol as the placed piece become empty.
    - Adjacent cells with the opposite symbol are converted to the player's symbol.
    - Empty cells remain unchanged.
    The game ends when no empty cells remain or when a player cannot make a move. The player who makes the last move wins. Moves are represented as coordinate pairs (row, column), e.g., (0, 1).
    """

    def _create_initial_board(self) -> np.ndarray:
        return np.zeros(self.board_size, dtype=int)

    def _get_legal_moves(self) -> List[Any]:
        """Returns all legal moves as a list of tuples."""
        legal_moves = []
        for i in range(self.board_size[0]):
            for j in range(self.board_size[1]):
                if self.board[i, j] == 0:
                    legal_moves.append((i, j))
        return legal_moves

    def make_move(self, move: Any) -> bool:
        """Executes a move and returns True if successful."""
        if move not in self._get_legal_moves():
            return False
        i, j = move
        self.board[i, j] = self.current_player
        # Flip adjacent cells
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for di, dj in directions:
            ni, nj = i + di, j + dj
            if 0 <= ni < self.board_size[0] and 0 <= nj < self.board_size[1]:
                if self.board[ni, nj] == self.current_player:
                    self.board[ni, nj] = 0
                elif self.board[ni, nj] == -self.current_player:
                    self.board[ni, nj] = self.current_player
        self.current_player *= -1
        return True

    def is_game_over(self) -> bool:
        """Returns True if no legal moves remain."""
        return len(self._get_legal_moves()) == 0

    def check_winner(self) -> Optional[Any]:
        """Returns the winner if the game is over."""
        if self.is_game_over():
            return -self.current_player  # Current player cannot move, previous player wins
        return None
