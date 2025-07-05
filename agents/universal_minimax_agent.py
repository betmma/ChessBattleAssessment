from typing import List, Tuple, Optional, Any
import random
import sys
import os
import psutil
from collections import OrderedDict

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.agent import Agent

class LRUCache:
    """LRU Cache implementation with automatic memory-based sizing"""
    
    def __init__(self, max_size: int):
        self.max_size = max_size
        self.cache = OrderedDict()
    
    def get(self, key):
        if key in self.cache:
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            return self.cache[key]
        return None
    
    def put(self, key, value):
        if key in self.cache:
            # Update existing key
            self.cache.move_to_end(key)
        else:
            # Add new key
            if len(self.cache) >= self.max_size:
                # Remove least recently used item
                self.cache.popitem(last=False)
        self.cache[key] = value
    
    def clear(self):
        self.cache.clear()

class UniversalMinimaxAgent(Agent):
    """Universal Minimax agent that works with any game implementing the Game interface"""
    
    def __init__(self, name: str = "UniversalMinimax", max_depth: int = 4):
        super().__init__(name)
        # Calculate cache size based on available memory
        cache_size = self._calculate_cache_size()
        self.score_cache = LRUCache(cache_size)
        # Maximum search depth
        self.max_depth = max_depth
    
    def _calculate_cache_size(self) -> int:
        """Calculate optimal cache size based on available memory"""
        try:
            # Get available memory in bytes
            available_memory = psutil.virtual_memory().available
            # Use 5% of available memory for cache, assuming ~100 bytes per cache entry
            cache_size = min(max(1000, available_memory // (100 * 20)), 50000)
            return cache_size
        except:
            # Fallback to conservative size if psutil fails
            return 10000
    
    def get_move(self, game) -> str:
        """
        Get best move using Minimax algorithm with alpha-beta pruning
        
        Args:
            game: Game object implementing the Game interface
            
        Returns:
            str: Best move as string
        """
        player_value = game.get_current_player()
        best_move = None
        legal_moves = game.get_legal_moves()

        if not legal_moves: 
            return 'No legal moves available'

        if player_value == 1:  # Player 1 (maximizing player)
            best_score = -float('inf')
            alpha = -float('inf')
            beta = float('inf')
            
            for move in legal_moves:
                temp_game = game.clone()
                temp_game.make_move(move)
                
                score = self.minimax_alpha_beta(temp_game, alpha, beta, False, 1)
                
                if score > best_score:
                    best_score = score
                    best_move = move
                    
                alpha = max(alpha, score)
                if beta <= alpha:
                    break  # Beta cutoff - pruning
        else:  # Player -1 (minimizing player)
            best_score = float('inf')
            alpha = -float('inf')
            beta = float('inf')
            
            for move in legal_moves:
                temp_game = game.clone()
                temp_game.make_move(move)
                
                score = self.minimax_alpha_beta(temp_game, alpha, beta, True, 1)
                
                if score < best_score:
                    best_score = score
                    best_move = move
                    
                beta = min(beta, score)
                if beta <= alpha:
                    break  # Alpha cutoff - pruning
        
        # If all moves have same score, choose randomly
        if best_move is None and legal_moves:
            best_move = random.choice(legal_moves)
            
        return str(best_move)
    
    def get_action_rewards(self, game) -> dict[str, float]:
        """
        Get reward values for every possible move from the current player's perspective.
        This can be used for training a reinforcement learning model.

        Args:
            game: Game object implementing the Game interface

        Returns:
            A dictionary mapping each legal move (as a string) to its minimax score.
        """
        player_value = game.get_current_player()
        legal_moves = game.get_legal_moves()
        rewards = {}

        if not legal_moves:
            return {}

        alpha = -float('inf')
        beta = float('inf')

        for move in legal_moves:
            temp_game = game.clone()
            temp_game.make_move(move)

            # The next player is the opposite of the current one.
            is_next_player_maximizing = (player_value == -1)

            # The score is always from the perspective of player 1.
            score = self.minimax_alpha_beta(temp_game, alpha, beta, is_next_player_maximizing, 1)

            # The reward for an action is the value of the resulting state from the current player's perspective.
            # If current player is 1, reward is the score.
            # If current player is -1, reward is the negative of the score.
            reward = score if player_value == 1 else -score
            rewards[str(move)] = reward

        return rewards
    
    def minimax_alpha_beta(self, game, alpha, beta, maximizing_player, depth):
        """
        Universal Minimax algorithm with alpha-beta pruning and depth limit
        
        Args:
            game: Game object implementing the Game interface
            alpha: Best value that maximizing player can guarantee so far
            beta: Best value that minimizing player can guarantee so far
            maximizing_player: True if current player is maximizing (player 1), False if minimizing (player -1)
            depth: Current search depth
            
        Returns:
            Score from player 1's perspective
        """
        # Create a hashable representation of the game state
        state_key = self._get_state_key(game, depth)
        
        # Check if this state has already been evaluated
        cached_result = self.score_cache.get(state_key)
        if cached_result is not None:
            return cached_result
        
        winner = game.check_winner()
        if winner is not None:
            if winner == 1: 
                score = 1000 - depth     # Player 1 wins (prefer shorter paths to victory)
            elif winner == -1: 
                score = -1000 + depth    # Player -1 wins (prefer longer paths to defeat)
            else:
                score = 0                # Draw
            
            # Cache the result
            self.score_cache.put(state_key, score)
            return score
        
        # If we've reached maximum depth, use the game's heuristic evaluation
        if depth >= self.max_depth:
            score = game.evaluate_position()
            self.score_cache.put(state_key, score)
            return score
        
        if maximizing_player:  # Player 1's turn, maximizing player
            max_eval = -float('inf')
            for move in game.get_legal_moves():
                temp_game = game.clone()
                temp_game.make_move(move)
                eval_score = self.minimax_alpha_beta(temp_game, alpha, beta, False, depth + 1)
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                
                # Alpha-beta pruning: if beta <= alpha, we can stop evaluating
                if beta <= alpha:
                    break  # Beta cutoff
            
            # Cache the result
            self.score_cache.put(state_key, max_eval)
            return max_eval
        else:  # Player -1's turn, minimizing player
            min_eval = float('inf')
            for move in game.get_legal_moves():
                temp_game = game.clone()
                temp_game.make_move(move)
                eval_score = self.minimax_alpha_beta(temp_game, alpha, beta, True, depth + 1)
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                
                # Alpha-beta pruning: if beta <= alpha, we can stop evaluating
                if beta <= alpha:
                    break  # Alpha cutoff
            
            # Cache the result
            self.score_cache.put(state_key, min_eval)
            return min_eval
    
    def _get_state_key(self, game, depth=0):
        """
        Create a hashable representation of the game state using only the Game interface
        
        Args:
            game: Game object implementing the Game interface
            depth: Current search depth
            
        Returns:
            A tuple representing the game state, current player, and depth
        """
        # Use the game's string representation as the state key
        # This is the most universal approach since all games implement get_state_representation
        state_str = game.get_state_representation()
        current_player = game.get_current_player()
        return (state_str, current_player, depth)