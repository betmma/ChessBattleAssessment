import copy
from typing import List, Tuple, Optional, Dict
import sys
import os
import re
import logging
from agents.agent import Agent
from agents.vllm_agent import VLLMAgent

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from games.game import Game

class NimGame(Game):
    """
    (Misère) Nim game implementation with 3 piles.
    Players take turns removing stones from any single pile.
    The player who takes the last stone loses.
    """
    
    def __init__(self, piles: List[int] = None):
        super().__init__()
        # Default to 3 piles with 3, 4, 5 stones respectively
        self.piles = piles if piles else self._random_init_()
        self.initial_piles = self.piles.copy()
        self.player_1_symbol = 'P1'
        self.player_2_symbol = 'P2'
        self._game_over_forced_forfeit = False
        self._setup_default_prompt()
        
    def _random_init_(self):
        """Randomly initialize the game state"""
        import random
        return [random.randint(1, 5) for _ in range(3)]
        
    def _setup_default_prompt(self):
        """Set default prompt template"""
        self.system_prompt = (
            "You are playing Nim. In this game, there are multiple piles of stones. "
            "On your turn, you must remove one or more stones from exactly ONE pile. "
            "The player who takes the LAST stone LOSES the game. "
            # "Think strategically about the Nim-sum (XOR of all pile sizes) to find the optimal move. "
            "After your thinking, provide your move as (pile_index, stones_to_remove). "
            "Example: To remove 2 stones from pile 0, respond with `(0,2)`. "
            "Your response format: `<think>Your reasoning...</think>(pile_index,stones_to_remove)`"
        )
        self.user_prompt_template = (
            "{board_representation}\n"
            "You are player '{player_symbol}'.\n"
            "Your available legal moves: [{legal_moves_str}]\n"
            "Provide your thinking and final move in the format: `<think>...</think>(pile_index,stones_to_remove)`"
        )
        self.system_prompt_no_thinking = (
            "You are playing Nim. Remove stones from exactly ONE pile on your turn. "
            "The player who takes the LAST stone LOSES. "
            "Your response MUST be your chosen move as (pile_index, stones_to_remove). "
            "Example: To remove 2 stones from pile 0, respond with `(0,2)`. "
            "Do not add any other text or explanation."
        )
        self.user_prompt_template_no_thinking = (
            "{board_representation}\n"
            "You are player '{player_symbol}'.\n"
            "Your available legal moves: [{legal_moves_str}]\n"
            "Choose your move (e.g., `(0,2)` to remove 2 stones from pile 0):"
        )
    
    def get_player_symbol(self, player_value):
        """Get the symbol representation for a player"""
        if player_value == 1: 
            return self.player_1_symbol
        if player_value == -1: 
            return self.player_2_symbol
        return "??"
    
    def get_legal_moves(self) -> List[Tuple[int, int]]:
        """Return all legal moves as (pile_index, stones_to_remove)"""
        moves = []
        for pile_idx, pile_size in enumerate(self.piles):
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
        if pile_idx < 0 or pile_idx >= len(self.piles):
            return False
        if stones_to_remove < 1 or stones_to_remove > self.piles[pile_idx]:
            return False
        
        # Execute move
        self.piles[pile_idx] -= stones_to_remove
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
        return all(pile == 0 for pile in self.piles)
    
    def get_current_player(self):
        """Get current player's turn"""
        return self.current_player
    
    def get_board_representation_for_llm(self) -> str:
        """Get board state representation for LLM"""
        representation = "Current game state:\n"
        for i, pile_size in enumerate(self.piles):
            stones_display = "●" * pile_size if pile_size > 0 else "(empty)"
            representation += f"Pile {i}: {stones_display} ({pile_size} stones)\n"
        
        # # Add Nim-sum for advanced players
        # nim_sum = 0
        # for pile in self.piles:
        #     nim_sum ^= pile
        # representation += f"Nim-sum (XOR): {nim_sum}\n"
        
        return representation
    
    def get_key_for_cache(self) -> tuple:
        """Get a unique key for caching game state, without current player or game over state"""
        return tuple(self.piles)
    
    def load_state_from_representation(self, state_str: str) -> bool:
        """Load game state from string representation"""
        try:
            lines = state_str.strip().split('\n')
            pile_lines = [line for line in lines if line.startswith('Pile ')]
            
            new_piles = []
            for line in pile_lines:
                # Extract pile size from line like "Pile 0: ●●● (3 stones)"
                match = re.search(r'\((\d+) stones\)', line)
                if match:
                    new_piles.append(int(match.group(1)))
                else:
                    return False
            
            if len(new_piles) != len(self.piles):
                return False
            
            self.piles = new_piles
            
            # Try to extract current player from subsequent lines
            for line in lines:
                if "Current turn:" in line:
                    if "P1" in line:
                        self.current_player = 1
                    elif "P2" in line:
                        self.current_player = -1
            
            return True
        except Exception:
            return False
    
    def _format_legal_moves_for_prompt(self, legal_moves: List[Tuple[int, int]]) -> str:
        """Format legal moves for display in prompt"""
        return ", ".join([f"({pile_idx},{stones})" for pile_idx, stones in legal_moves])
    
    def parse_move_from_output(self, raw_output: str, legal_moves: List[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
        """Parse a move from agent's output string, validating against legal moves"""
        # Try to find patterns like (0,2) or [0,2]
        patterns = [
            r'\((\d+),\s*(\d+)\)',  # (0,2)
            r'\[(\d+),\s*(\d+)\]',  # [0,2]
            r'(\d+),\s*(\d+)',      # 0,2
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, raw_output)
            if matches:
                try:
                    pile_idx, stones = int(matches[0][0]), int(matches[0][1])
                    move = (pile_idx, stones)
                    if move in legal_moves:
                        return move
                except (ValueError, IndexError):
                    continue
        
        return None
    
    def reset(self) -> None:
        """Reset the game to initial state"""
        self.piles = self.initial_piles.copy()
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
        large_piles = [pile for pile in self.piles if pile > 1]
        single_piles = [pile for pile in self.piles if pile == 1]
        
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
            for pile in self.piles:
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
        cloned.piles = self.piles.copy()
        cloned.current_player = self.current_player
        cloned._game_over_forced_forfeit = self._game_over_forced_forfeit
        return cloned
