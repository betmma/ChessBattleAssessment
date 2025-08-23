import random
import re
import numpy as np
from typing import List, Tuple, Optional, Any
from games.board_game import BoardGame


class NimGame(BoardGame, board_size=(4,), move_arity=2):
    """
    (Misère) Nim game implementation with 3-4 piles.
    Players take turns removing stones from any single pile.
    The player who takes the last stone loses.
    """
    
    name = "Nim"
    game_introduction = (
        "Nim is a game where players take turns removing stones from piles. "
        "On your turn, you must remove one or more stones from exactly ONE pile. "
        "The player who takes the LAST stone LOSES the game (misère version). "
        "The game state shows each pile with its number of stones. "
        "Moves are in the format (pile_index, stones_to_remove), where pile_index "
        "starts from 0 and stones_to_remove is the number of stones to take from that pile."
    )
    
    player_symbols = {1: 'P1', -1: 'P2', 0: '.'}
    
    def __init__(self, piles: List[int] = None):
        # Initialize with random piles if not provided
        if piles is None:
            piles = self._random_init_()
        
        # Ensure we have exactly 4 elements for the board (pad with zeros if needed)
        while len(piles) < 4:
            piles.append(0)
        
        self.initial_piles = piles[:4]  # Only take first 4 piles
        self._game_over_forced_forfeit = False
        
        super().__init__()
    
    def _random_init_(self) -> List[int]:
        """Randomly initialize the game state"""
        num_piles = random.randint(3, 4)
        return [random.randint(1, 7) for _ in range(num_piles)]
    
    def _create_initial_board(self) -> np.ndarray:
        """Creates the initial board configuration."""
        return np.array(self.initial_piles, dtype=int)
    
    def get_board_representation_for_llm(self) -> str:
        """Get board state representation for LLM"""
        representation = "Current game state:\n"
        for i, pile_size in enumerate(self.board):
            if pile_size > 0:
                stones_display = "." * pile_size
                representation += f"Pile {i}: {stones_display} ({pile_size} stones)\n"
            else:
                representation += f"Pile {i}: (empty) (0 stones)\n"
        return representation
    
    def _get_legal_moves(self) -> List[Tuple[int, int]]:
        """Return all legal moves as (pile_index, stones_to_remove)"""
        moves = []
        for pile_idx, pile_size in enumerate(self.board):
            for stones in range(1, pile_size + 1):
                moves.append((pile_idx, stones))
        return moves
    
    def make_move(self, move: Tuple[int, int]) -> bool:
        """Execute a move, return True if move was legal and successful"""
        if not isinstance(move, (tuple, list)) or len(move) != 2:
            return False
        
        pile_idx, stones_to_remove = move
        
        # Validate move
        if not isinstance(pile_idx, int) or not isinstance(stones_to_remove, int):
            return False
        if pile_idx < 0 or pile_idx >= len(self.board):
            return False
        if stones_to_remove < 1 or stones_to_remove > self.board[pile_idx]:
            return False
        
        # Execute move
        self.board[pile_idx] -= stones_to_remove
        self.current_player *= -1  # Switch player (1 -> -1, -1 -> 1)
        return True
    
    def check_winner(self) -> Optional[int]:
        """Check for a winner, return winner player, 0 for draw, None if game continues"""
        if self._game_over_forced_forfeit:
            # Current player forfeited, so the other player wins
            return -self.current_player
        
        if self.is_game_over():
            # In misere Nim, the player who takes the last stone loses
            # The current player is the one who DIDN'T take the last stone
            # So the current player is the winner
            return self.current_player
        return None
    
    def is_game_over(self) -> bool:
        """Check if the game is over (all piles are empty)"""
        return np.all(self.board == 0)
    
    def get_key_for_cache(self) -> tuple:
        """Get a unique key for caching game state"""
        return (self.board.tobytes(),)
    
    def load_state_from_representation(self, state_str: str) -> bool:
        """Load game state from string representation"""
        try:
            lines = state_str.strip().split('\n')
            pile_lines = [line for line in lines if line.startswith('Pile ')]
            
            new_piles = [0] * 4  # Initialize with 4 piles
            for line in pile_lines:
                # Extract pile index and size from line like "Pile 0: ●●● (3 stones)"
                pile_match = re.search(r'Pile (\d+):', line)
                size_match = re.search(r'\((\d+) stones\)', line)
                
                if pile_match and size_match:
                    pile_idx = int(pile_match.group(1))
                    pile_size = int(size_match.group(1))
                    if 0 <= pile_idx < 4:
                        new_piles[pile_idx] = pile_size
            
            self.board = np.array(new_piles, dtype=int)
            
            # Try to extract current player from subsequent lines
            for line in lines:
                match = re.search(r"Current turn: . \(plays as (-?\d+)\)", line)
                if match:
                    self.current_player = int(match.group(1))
                    break
            
            return True
        except Exception:
            return False
    
    def reset(self) -> None:
        """Reset the game to initial state"""
        self.board = np.array(self.initial_piles, dtype=int)
        self.current_player = 1
        self._game_over_forced_forfeit = False
    
    def force_forfeit(self) -> None:
        """Force the current player to forfeit the game"""
        self._game_over_forced_forfeit = True
    
    def evaluate_position(self) -> float:
        """
        Evaluate the current position from player 1's perspective.
        In misère Nim, strategy differs from normal Nim.
        """
        if self._game_over_forced_forfeit:
            # Current player forfeited, so the other player wins
            winner = -self.current_player
            return 100.0 if winner == 1 else -100.0
        
        if self.is_game_over():
            # Game is over, winner is the current player (who didn't take last stone)
            winner = self.current_player
            return 100.0 if winner == 1 else -100.0
        
        # Count piles with more than 1 stone
        large_piles = [pile for pile in self.board if pile > 1]
        single_piles = [pile for pile in self.board if pile == 1]
        
        # Determine if current player is in winning or losing position
        current_player_winning = False
        
        # Misère Nim strategy:
        # 1. If there are no piles with >1 stones, count single piles
        if not large_piles:
            # Only single-stone piles remain
            # Player to move loses if there's an odd number of piles
            current_player_winning = (len(single_piles) % 2 == 0)
        
        # 2. If there are piles with >1 stones, use modified Nim-sum strategy
        elif len(large_piles) == 1:
            # Exactly one pile with >1 stones
            # Current player can control parity of 1-piles, so they're winning
            current_player_winning = True
        
        # 3. Multiple large piles: use standard Nim-sum but with caution near endgame
        else:
            nim_sum = 0
            for pile in self.board:
                nim_sum ^= pile
            
            # In misère Nim, Nim-sum strategy applies until near the end
            # Nim-sum != 0 means current player is in winning position
            current_player_winning = (nim_sum != 0)
        
        # Convert to Player 1's perspective
        if current_player_winning:
            # Current player is winning
            return 10.0 if self.current_player == 1 else -10.0
        else:
            # Current player is losing
            return -10.0 if self.current_player == 1 else 10.0
    
    def clone(self):
        """Create a deep copy of the current game state"""
        cloned = NimGame(self.initial_piles.copy())
        cloned.board = np.copy(self.board)
        cloned.current_player = self.current_player
        cloned._game_over_forced_forfeit = self._game_over_forced_forfeit
        return cloned
