
import numpy as np
from typing import List, Any, Optional, Tuple

from games.board_game import BoardGame


class Capture(BoardGame, board_size=(5, 5), move_arity=2):
    """
    Capture is a 2-player board game on a 5x5 grid.
    - Players take turns placing their piece on an empty square.
    - If a player places a piece that sandwiches one or more of the opponent's pieces
      between two of their own pieces (horizontally, vertically, or diagonally),
      the opponent's pieces are captured and flipped.
    - The game ends when the board is full. The player with the most pieces wins.
    """
    name = "Capture"
    game_introduction = """
    This is Capture, a 2-player board game on a 5x5 grid.
    Players take turns placing their piece ('X' or 'O') on an empty square.
    A move consists of placing a piece on an empty square.
    If your move sandwiches one or more of your opponent's pieces between two of your own pieces, you capture them.
    Captures can happen horizontally, vertically, or diagonally.
    The game ends when the board is full. The player with the most pieces on the board wins.
    A move is represented by a tuple (row, col), where row and col are 0-indexed.
    For example, (0, 0) is the top-left corner, and (4, 4) is the bottom-right corner.
    """

    def get_legal_moves(self) -> List[Any]:
        """Return all currently legal moves."""
        moves = []
        for r in range(self.board_size[0]):
            for c in range(self.board_size[1]):
                if self.board[r, c] == 0:
                    moves.append((r, c))
        return moves

    def make_move(self, move: Any) -> bool:
        """Executes a move."""
        if move not in self.get_legal_moves():
            return False

        r, c = move
        self.board[r, c] = self.current_player

        # Check for captures
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                self._check_and_capture(r, c, dr, dc)

        self.current_player *= -1
        return True

    def _check_and_capture(self, r: int, c: int, dr: int, dc: int):
        """Check for captures in a specific direction and perform them."""
        opponent = -self.current_player
        line = []
        for i in range(1, self.board_size[0]):
            nr, nc = r + i * dr, c + i * dc
            if not (0 <= nr < self.board_size[0] and 0 <= nc < self.board_size[1]):
                break
            if self.board[nr, nc] == self.current_player:
                # Found a sandwich
                for captured_r, captured_c in line:
                    self.board[captured_r, captured_c] = self.current_player
                break
            elif self.board[nr, nc] == opponent:
                line.append((nr, nc))
            else: # Empty square
                break

    def check_winner(self) -> Optional[Any]:
        """Checks for a winner."""
        if not self.is_game_over():
            return None

        player1_count = np.sum(self.board == 1)
        player2_count = np.sum(self.board == -1)

        if player1_count > player2_count:
            return 1
        elif player2_count > player1_count:
            return -1
        else:
            return 0  # Draw

    def is_game_over(self) -> bool:
        """Checks if the game is over."""
        return not np.any(self.board == 0)

    def evaluate_position(self) -> float:
        """
        Evaluates the current board position for the minimax agent.
        The evaluation is the difference in the number of pieces.
        """
        return np.sum(self.board)
