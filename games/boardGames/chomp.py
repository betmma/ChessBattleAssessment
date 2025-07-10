import numpy as np
from typing import List, Any, Optional, Tuple

from games.board_game import BoardGame

class Chomp(BoardGame, board_size=(4, 7), move_arity=2):
    """
    An implementation of the Poisoned Chocolate (Chomp) game.
    """
    name = "Chomp"
    game_introduction = (
        "Chomp is a 2-player game played on a rectangular chocolate bar. "
        "Players take turns choosing a block (r, c) and 'eating' it, which removes the block at (r, c) and all blocks to its right and below (i.e., all blocks (i, j) where i >= r and j >= c). "
        "The top-left block at (0, 0) is poisoned. The player who is forced to eat the poisoned block loses the game."
    )

    def __init__(self, rows: int = 4, cols: int = 7):
        """
        Initializes the Chomp game.

        Args:
            rows: The number of rows in the chocolate bar.
            cols: The number of columns in the chocolate bar.
        """
        self.board_size = (rows, cols)
        super().__init__()

    def _create_initial_board(self) -> np.ndarray:
        """
        Creates and returns the initial board configuration with all chocolate blocks present.
        1 represents an uneaten block, 0 represents an eaten block.
        """
        return np.ones(self.board_size, dtype=int)

    def _get_legal_moves(self) -> List[Tuple[int, int]]:
        """
        A legal move is to select any uneaten block.
        """
        return [(r, c) for r in range(self.board_size[0]) for c in range(self.board_size[1]) if self.board[r, c] == 1]

    def make_move(self, move: Tuple[int, int]) -> bool:
        """
        Executes a move by 'eating' the selected block and all blocks to its right and below.
        A move is a tuple (row, col).
        """
        r, c = move
        if not (0 <= r < self.board_size[0] and 0 <= c < self.board_size[1]):
            return False  # Move out of bounds
        if self.board[r, c] == 0:
            return False  # Block already eaten

        # Eat the block and all blocks to the right and below
        for i in range(r, self.board_size[0]):
            for j in range(c, self.board_size[1]):
                self.board[i, j] = 0
        
        self.current_player *= -1
        return True

    def is_game_over(self) -> bool:
        """
        The game is over if the poisoned block at (0, 0) is eaten.
        """
        return self.board[0, 0] == 0

    def check_winner(self) -> Optional[int]:
        """
        Checks for a winner. The player who eats the poisoned block at (0,0) loses.
        So the winner is the other (current) player.
        """
        if self.is_game_over():
            return self.current_player
        return None

    def evaluate_position(self) -> float:
        """
        Evaluates the current board position for player 1.
        """
        player = self.current_player
        value = 100
        win = None
        
        # find if remaining block is rectangle
        remaining_blocks = np.argwhere(self.board == 1)
        max_row = remaining_blocks[:, 0].max()+1 if remaining_blocks.size > 0 else -1
        max_col = remaining_blocks[:, 1].max()+1 if remaining_blocks.size > 0 else -1
        ones=remaining_blocks.size//2
        if max_row*max_col == ones:
            if max_row==max_col>1 or max_row==2 or max_col==2 or max_row==1!=max_col or max_row!=1==max_col:
                win = True
            elif max_row==max_col==1: win = False
        elif max_row+max_col-1 == ones: # L shape
            if max_row==max_col: win = False
            else: win = True
        #print(self.board,win,max_row,max_col,ones)
        if win is True:
            return value * player
        elif win is False:
            return -value * player
        return 0
