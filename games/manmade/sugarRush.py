import numpy as np
from typing import List, Any, Optional, Tuple
from collections import deque

from games.board_game import BoardGame

class SugarRush(BoardGame, board_size=(5, 5), move_arity=2):
    """
    Sugar Rush - A strategic crystal collection game.
    
    Players take turns placing sugar crystals on a 5x5 board.
    The goal is to create the largest connected group of crystals.
    Some squares contain "bitter spots" that block placement.
    """
    name = "Sugar Rush"
    game_introduction = (
        "Sugar Rush is a 2-player strategy game played on a 5x5 grid. "
        "Players take turns placing their sugar crystals (X for player 1, O for player -1) on empty squares. "
        "Some squares contain bitter spots (B) that cannot be used. "
        "The objective is to create the largest connected group of your crystals. "
        "Connected means adjacent horizontally or vertically (not diagonally). "
        "The game ends when the board is full or no legal moves remain. "
        "The player with the largest connected group of crystals wins. "
        "In case of a tie, the last player to move wins."
    )

    def __init__(self):
        """
        Initializes the Sugar Rush game with a 5x5 board containing some bitter spots.
        """
        super().__init__()
        self.moves_made = 0
        self.last_player = None
        
    def _create_initial_board(self) -> np.ndarray:
        """
        Creates the initial board with some bitter spots (-2) for strategic complexity.
        0 = empty, 1 = player 1 crystal, -1 = player -1 crystal, -2 = bitter spot
        """
        board = np.zeros(self.board_size, dtype=int)
        
        # Place bitter spots in strategic positions to create interesting gameplay
        # These positions are chosen to create tactical decisions
        bitter_spots = [(1, 1), (1, 3), (3, 1), (3, 3)]
        
        for r, c in bitter_spots:
            board[r, c] = -2
            
        return board

    def _get_legal_moves(self) -> List[Tuple[int, int]]:
        """
        Returns all empty squares (value 0) as legal moves.
        """
        legal_moves = []
        for r in range(self.board_size[0]):
            for c in range(self.board_size[1]):
                if self.board[r, c] == 0:
                    legal_moves.append((r, c))
        return legal_moves

    def make_move(self, move: Tuple[int, int]) -> bool:
        """
        Places a sugar crystal at the specified position.
        """
        r, c = move
        
        # Check bounds
        if not (0 <= r < self.board_size[0] and 0 <= c < self.board_size[1]):
            return False
            
        # Check if square is empty
        if self.board[r, c] != 0:
            return False
            
        # Place the crystal
        self.board[r, c] = self.current_player
        self.moves_made += 1
        self.last_player = self.current_player
        
        # Switch players
        self.current_player *= -1
        
        return True

    def is_game_over(self) -> bool:
        """
        Game ends when no legal moves remain.
        """
        return len(self._get_legal_moves()) == 0

    def check_winner(self) -> Optional[int]:
        """
        Determines the winner by finding the largest connected group.
        Returns the player with the largest connected group.
        In case of tie, the last player to move wins.
        """
        if not self.is_game_over():
            return None
            
        player1_max = self._get_largest_connected_group(1)
        player2_max = self._get_largest_connected_group(-1)
        
        if player1_max > player2_max:
            return 1
        elif player2_max > player1_max:
            return -1
        else:
            # Tie - last player to move wins
            return self.last_player

    def _get_largest_connected_group(self, player: int) -> int:
        """
        Uses BFS to find the largest connected group of crystals for a player.
        """
        visited = np.zeros(self.board_size, dtype=bool)
        max_group_size = 0
        
        for r in range(self.board_size[0]):
            for c in range(self.board_size[1]):
                if self.board[r, c] == player and not visited[r, c]:
                    group_size = self._bfs_group_size(r, c, player, visited)
                    max_group_size = max(max_group_size, group_size)
                    
        return max_group_size

    def _bfs_group_size(self, start_r: int, start_c: int, player: int, visited: np.ndarray) -> int:
        """
        Performs BFS to count the size of a connected group starting from (start_r, start_c).
        """
        queue = deque([(start_r, start_c)])
        visited[start_r, start_c] = True
        size = 0
        
        # Directions: up, down, left, right
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        while queue:
            r, c = queue.popleft()
            size += 1
            
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                
                if (0 <= nr < self.board_size[0] and 
                    0 <= nc < self.board_size[1] and 
                    not visited[nr, nc] and 
                    self.board[nr, nc] == player):
                    
                    visited[nr, nc] = True
                    queue.append((nr, nc))
                    
        return size

    def evaluate_position(self) -> float:
        """
        Evaluates the current position for the minimax algorithm.
        Considers current largest groups and potential for expansion.
        """
        if self.is_game_over():
            winner = self.check_winner()
            if winner == 1:
                return 1000.0
            elif winner == -1:
                return -1000.0
            else:
                return 0.0
        
        # Evaluate based on current largest groups
        player1_max = self._get_largest_connected_group(1)
        player2_max = self._get_largest_connected_group(-1)
        
        # Base score from group sizes
        score = (player1_max - player2_max) * 10
        
        # Add bonus for having more crystals overall
        player1_count = np.sum(self.board == 1)
        player2_count = np.sum(self.board == -1)
        score += (player1_count - player2_count) * 2
        
        # Add bonus for potential expansion opportunities
        score += self._evaluate_expansion_potential(1) * 3
        score -= self._evaluate_expansion_potential(-1) * 3
        
        return score

    def _evaluate_expansion_potential(self, player: int) -> float:
        """
        Evaluates how many empty squares are adjacent to the player's crystals.
        """
        potential = 0
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        for r in range(self.board_size[0]):
            for c in range(self.board_size[1]):
                if self.board[r, c] == player:
                    for dr, dc in directions:
                        nr, nc = r + dr, c + dc
                        if (0 <= nr < self.board_size[0] and 
                            0 <= nc < self.board_size[1] and 
                            self.board[nr, nc] == 0):
                            potential += 1
                            
        return potential

    def get_board_representation_for_llm(self) -> str:
        """
        Returns a string representation of the board for the LLM.
        Uses X for player 1, O for player -1, . for empty, B for bitter spots.
        """
        symbols = {1: 'X', -1: 'O', 0: '.', -2: 'B'}
        return "\n".join([" ".join([symbols[cell] for cell in row]) for row in self.board])

    def reset(self) -> None:
        """
        Resets the game to its initial state.
        """
        super().reset()
        self.moves_made = 0
        self.last_player = None

    def clone(self):
        """
        Creates a deep copy of the game state.
        """
        new_game = self.__class__()
        new_game.board = np.copy(self.board)
        new_game.current_player = self.current_player
        new_game.moves_made = self.moves_made
        new_game.last_player = self.last_player
        return new_game
