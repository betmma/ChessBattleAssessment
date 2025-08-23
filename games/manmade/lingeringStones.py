import numpy as np
from typing import List, Any, Optional, Tuple
from games.board_game import BoardGame

class LingeringStonesGame(BoardGame, board_size=(4, 4), move_arity=4):
    """
    A strategic board game where moving a piece leaves a permanent 'lingering stone',
    blocking the square it came from. The objective is to trap the opponent,
    leaving them with no legal moves.
    """

    name = "Lingering Stones"
    game_introduction = (
        "This is Lingering Stones, a game of strategy and attrition on a 4x4 board. "
        "Each player controls 3 pieces. On your turn, you must move one of your pieces "
        "to an adjacent (horizontal or vertical) empty square. The square you move "
        "FROM becomes a permanent 'lingering stone' (#), which cannot be moved to or "
        "through for the rest of the game. You win by trapping your opponent so they "
        "have no legal moves. The game ends in a draw after 40 total moves. "
        "Move format: (from_row, from_col, to_row, to_col). "
    )

    player_symbols = {1: 'X', -1: 'O', 0: '.', 2: '#'}  # 2 for lingering stones

    def __init__(self):
        """Initializes the Lingering Stones game."""
        super().__init__()
        self.move_count = 0

    def _create_initial_board(self) -> np.ndarray:
        """
        Creates the initial 4x4 board with pieces in their starting positions.
        Player 1 (X) starts at the top. Player 2 (O) starts at the bottom.
        """
        board = np.zeros(self.board_size, dtype=int)
        # Player 1 pieces
        board[0, 0] = board[0, 1] = board[0, 2] = 1
        # Player 2 pieces
        board[3, 0] = board[3, 1] = board[3, 2] = -1
        return board

    def _get_legal_moves(self) -> List[Tuple[int, int, int, int]]:
        """
        Calculates all legal moves for the current player.
        A move is legal if it's from a player's piece to an adjacent (up, down,
        left, or right) empty square.
        """
        moves = []
        piece_locations = np.argwhere(self.board == self.current_player)
        for r, c in piece_locations:
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.board_size[0] and 0 <= nc < self.board_size[1] and self.board[nr, nc] == 0:
                    moves.append((r, c, nr, nc))
        return moves

    def make_move(self, move: Tuple[int, int, int, int]) -> bool:
        """
        Executes a move, updates the board, and switches the current player.
        The original square of the piece becomes a lingering stone.
        """
        from_r, from_c, to_r, to_c = move
        if move not in self.get_legal_moves():
            return False
        
        self.board[to_r, to_c] = self.current_player
        self.board[from_r, from_c] = 2  # Leave a lingering stone
        self.current_player *= -1
        self.move_count += 1
        return True

    def check_winner(self) -> Optional[int]:
        """
        Checks if there is a winner. A player wins if the opponent has no legal moves.
        """
        if not self.get_legal_moves():
            # The current player has no moves, so the other player wins.
            return self.current_player * -1
        return None

    def is_game_over(self) -> bool:
        """
        Checks if the game is over, either by a win or a draw.
        A draw occurs if 40 total moves have been made.
        """
        if self.check_winner() is not None:
            return True
        if self.move_count >= 40:
            return True
        return False

    def evaluate_position(self) -> float:
        """
        Provides a simple evaluation of the current board position for the Minimax agent.
        The evaluation is the number of legal moves for the current player minus the
        number of legal moves for the opponent.
        """
        my_moves = len(self.get_legal_moves())
        
        # Temporarily switch player to evaluate opponent's moves
        self.current_player *= -1
        opponent_moves = len(self.get_legal_moves())
        self.current_player *= -1 # Switch back

        return float(my_moves - opponent_moves)

    def reset(self) -> None:
        """Resets the game to its initial state."""
        super().reset()
        self.move_count = 0

    def clone(self):
        """Creates a deep copy of the game state."""
        new_game = super().clone()
        new_game.move_count = self.move_count
        return new_game
