from typing import List, Tuple, Optional, Any
import random
import sys
import os
import psutil
from collections import OrderedDict
import threading

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.agent import Agent

class LRUCache:
    """LRU Cache implementation with automatic memory-based sizing"""
    
    def __init__(self, max_size: int):
        self.max_size = max_size
        self.cache = OrderedDict()
        self._lock = threading.Lock()
    
    def get(self, key):
        with self._lock:
            if key in self.cache:
                # Move to end (most recently used)
                self.cache.move_to_end(key)
                return self.cache[key]
            return None
    
    def put(self, key, value):
        with self._lock:
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
        with self._lock:
            self.cache.clear()

class CacheManager:
    """Singleton cache manager that maintains one cache per game type"""
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(CacheManager, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not getattr(self, '_initialized', False):
            self._game_caches = {}
            self._cache_lock = threading.Lock()
            self._initialized = True
    
    def get_cache_for_game(self, game_class_name: str, cache_size: int = None) -> LRUCache:
        """Get or create a cache for a specific game type"""
        with self._cache_lock:
            if game_class_name not in self._game_caches:
                if cache_size is None:
                    cache_size = self._calculate_default_cache_size()
                self._game_caches[game_class_name] = LRUCache(cache_size)
            return self._game_caches[game_class_name]
    
    def clear_cache_for_game(self, game_class_name: str):
        """Clear cache for a specific game type"""
        with self._cache_lock:
            if game_class_name in self._game_caches:
                self._game_caches[game_class_name].clear()
    
    def clear_all_caches(self):
        """Clear all caches"""
        with self._cache_lock:
            for cache in self._game_caches.values():
                cache.clear()
            self._game_caches.clear()
    
    def _calculate_default_cache_size(self) -> int:
        """Calculate optimal cache size based on available memory"""
        try:
            # Get available memory in bytes
            available_memory = psutil.virtual_memory().available
            # Use 5% of available memory for cache, assuming ~100 bytes per cache entry
            cache_size = min(max(1000, available_memory // (100 * 20)), 50000000)
            return cache_size
        except:
            # Fallback to conservative size if psutil fails
            return 10000

class UniversalMinimaxAgent(Agent):
    """Universal Minimax agent that works with any game implementing the Game interface"""
    
    def __init__(self, name: str = "UniversalMinimax", max_depth: int = 4, debug: bool = False, same_return_random: bool = True):
        super().__init__(name)
        # Use singleton cache manager
        self.cache_manager = CacheManager()
        # We'll get the actual cache when we first encounter a game
        self.score_cache = None
        self.current_game_class = None
        # Maximum search depth
        self.max_depth = max_depth
        # Debug mode (default: False for production)
        self.debug = debug
        # Option to disable caching for debugging
        self.use_cache = True
        # Option to return random move if multiple moves have the same score
        self.same_return_random = same_return_random
    
    def _get_game_cache(self, game):
        """Get or create the cache for the current game type"""
        game_class_name = game.__class__.__name__
        if self.current_game_class != game_class_name:
            self.current_game_class = game_class_name
            cache_size = self._calculate_cache_size()
            self.score_cache = self.cache_manager.get_cache_for_game(game_class_name, cache_size)
        return self.score_cache
        # Debug mode (default: False for production)
        self.debug = debug
        # Option to disable caching for debugging
        self.use_cache = True
        # Option to return random move if multiple moves have the same score
        self.same_return_random = same_return_random
    
    def set_debug_mode(self, debug: bool):
        """Enable or disable debug mode"""
        self.debug = debug
    
    def disable_cache(self):
        """Disable caching for debugging purposes"""
        self.use_cache = False
        if self.score_cache:
            self.score_cache.clear()
    
    def enable_cache(self):
        """Re-enable caching"""
        self.use_cache = True
    
    def clear_cache_for_current_game(self):
        """Clear cache for the current game type"""
        if self.current_game_class:
            self.cache_manager.clear_cache_for_game(self.current_game_class)
    
    def _calculate_cache_size(self) -> int:
        """Calculate optimal cache size based on available memory"""
        try:
            # Get available memory in bytes
            available_memory = psutil.virtual_memory().available
            # Use 5% of available memory for cache, assuming ~100 bytes per cache entry
            cache_size = min(max(1000, available_memory // (100 * 20)), 50000000)
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
        # Ensure we have the correct cache for this game type
        self._get_game_cache(game)
        
        rewards = self.get_action_rewards(game)
        if self.same_return_random:
            best_moves,best_reward=[], None
            for move, reward in rewards.items():
                if best_reward is None or reward > best_reward:
                    best_moves = [move]
                    best_reward = reward
                elif reward == best_reward:
                    best_moves.append(move)
            if best_moves:
                return random.choice(best_moves)
        best_move = max(rewards, key=rewards.get, default=None)
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
        # Ensure we have the correct cache for this game type
        self._get_game_cache(game)
        
        from utils.safe_json_dump import clean_np_types
        player_value = game.get_current_player()
        legal_moves = game.get_legal_moves()
        rewards = {}

        if not legal_moves:
            return {}

        for move in legal_moves:
            temp_game = game.clone()
            temp_game.make_move(move)

            # The next player is the opposite of the current one.
            is_next_player_maximizing = (player_value == -1)

            # The score is always from the perspective of player 1.
            score = self.minimax_alpha_beta(temp_game, -float('inf'), float('inf'), is_next_player_maximizing, 1)

            # The reward for an action is the value of the resulting state from the current player's perspective.
            # If current player is 1, reward is the score.
            # If current player is -1, reward is the negative of the score.
            reward = score if player_value == 1 else -score
            rewards[str(clean_np_types(move))] = reward

        return rewards
    
    def minimax_alpha_beta(self, game, alpha, beta, maximizing_player, depth):
        """
        Universal Minimax algorithm with alpha-beta pruning and sophisticated caching
        
        Args:
            game: Game object implementing the Game interface
            alpha: Best value that maximizing player can guarantee so far
            beta: Best value that minimizing player can guarantee so far
            maximizing_player: True if current player is maximizing (player 1), False if minimizing (player -1)
            depth: Current search depth
            
        Returns:
            Score from player 1's perspective
        """
        # Store original alpha-beta bounds for cache validation
        original_alpha, original_beta = alpha, beta
        
        # Create a hashable representation of the game state
        # Include depth to properly distinguish positions at different search depths
        state_key = self._get_state_key(game, depth, include_depth=True)
        
        if self.debug:
            if hasattr(game, 'piles'):
                print(f"Debug depth {depth}: piles={game.piles}, player={game.get_current_player()}, maximizing={maximizing_player}, state_key={state_key}")
        
        # Check game status first
        winner = game.check_winner()
        
        # Check if this state has already been evaluated
        # Use cache for terminal positions, heuristic evaluations, and safe intermediate results
        cached_result = None
        if self.use_cache and (winner is not None or depth >= self.max_depth):
            cached_result = self.score_cache.get(state_key)
        
        if cached_result is not None:
            if self.debug:
                print(f"Debug depth {depth}: Cache HIT! Returning {cached_result}")
            return cached_result
        
        
        if winner is not None:
            if winner == 1: 
                score = 1000 - depth     # Player 1 wins (prefer shorter paths to victory)
            elif winner == -1: 
                score = -1000 + depth    # Player -1 wins (prefer longer paths to defeat)
            else:
                score = 0                # Draw
            
            if self.debug:
                print(f"Debug depth {depth}: Terminal state! Winner={winner}, Score={score}")
            
            # Cache the result for terminal states
            if self.use_cache:
                self.score_cache.put(state_key, score)
            return score
        
        # If we've reached maximum depth, use the game's heuristic evaluation
        if depth >= self.max_depth:
            score = game.evaluate_position()
            if self.debug:
                print(f"Debug depth {depth}: Max depth reached! Heuristic score={score}")
            if self.use_cache:
                self.score_cache.put(state_key, score)
            return score
        
        # Track if alpha-beta cutoff occurred
        cutoff_occurred = False
        
        if maximizing_player:  # Player 1's turn, maximizing player
            max_eval = -float('inf')
            for move in game.get_legal_moves():
                temp_game = game.clone()
                temp_game.make_move(move)
                eval_score = self.minimax_alpha_beta(temp_game, alpha, beta, False, depth + 1)
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                
                if self.debug:
                    print(f"Debug depth {depth}: Maximizing move {move}, score={eval_score}, max_eval={max_eval}")
                
                # Alpha-beta pruning: if beta <= alpha, we can stop evaluating
                if beta <= alpha:
                    if self.debug:
                        print(f"Debug depth {depth}: Alpha-beta cutoff! beta={beta} <= alpha={alpha}")
                    cutoff_occurred = True
                    break  # Beta cutoff
            
            if self.debug:
                print(f"Debug depth {depth}: Final maximizing result={max_eval}")
            
            # Cache intermediate results only if no cutoff occurred and bounds are wide enough
            # This ensures we computed the true minimax value, not just a bound
            if self.use_cache and not cutoff_occurred and (original_beta - original_alpha) >= 100:
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
                
                if self.debug:
                    print(f"Debug depth {depth}: Minimizing move {move}, score={eval_score}, min_eval={min_eval}")
                
                # Alpha-beta pruning: if beta <= alpha, we can stop evaluating
                if beta <= alpha:
                    if self.debug:
                        print(f"Debug depth {depth}: Alpha-beta cutoff! beta={beta} <= alpha={alpha}")
                    cutoff_occurred = True
                    break  # Alpha cutoff
            
            if self.debug:
                print(f"Debug depth {depth}: Final minimizing result={min_eval}")
            
            # Cache intermediate results only if no cutoff occurred and bounds are wide enough
            if self.use_cache and not cutoff_occurred and (original_beta - original_alpha) >= 100:
                self.score_cache.put(state_key, min_eval)
            
            return min_eval
    
    def _get_state_key(self, game, depth=0, include_depth=False):
        """
        Create a hashable representation of the game state using only the Game interface
        
        Args:
            game: Game object implementing the Game interface
            depth: Current search depth (only included if include_depth=True)
            include_depth: Whether to include depth in the cache key
            
        Returns:
            A tuple representing the game state, current player, and optionally depth
        """
        state_str = game.get_key_for_cache()
        current_player = game.get_current_player()
        if include_depth:
            return (state_str, current_player, depth)
        else:
            return (state_str, current_player)