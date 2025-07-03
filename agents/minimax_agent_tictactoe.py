from typing import List, Tuple, Optional, Any
import sys
import os
import random

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents import Agent

class MinimaxAgentTicTacToe(Agent):
    """Agent using Minimax algorithm"""
    
    def __init__(self, name: str = "MinimaxAgent"):
        super().__init__(name)
        # Cache for storing evaluated positions
        self.score_cache = {}
    
    def get_move(self, game) -> str:
        """
        Get best move using Minimax algorithm
        
        Args:
            game: Game object
            
        Returns:
            str: Best move as string
        """
        player_value = game.get_current_player()
        best_score = -float('inf') if player_value == 1 else float('inf')
        best_move = None
        legal_moves = game.get_legal_moves()

        if not legal_moves: 
            return 'No legal moves available'
        
        # Optimize performance - random first move
        if len(legal_moves) == 9:  
            return str(random.choice(legal_moves))

        # Clear cache for each new move decision to prevent memory growth
        # While keeping it during the minimax recursion
        self.score_cache = {}

        for move in legal_moves:
            temp_game = game.clone()
            temp_game.make_move(move)  # Player switches in temp_game
            
            # Score from X (value 1) perspective
            score = self.minimax(temp_game)
            
            if player_value == 1:  # X (maximizing player)
                if score > best_score:
                    best_score = score
                    best_move = move
            else:  # O (minimizing player)
                if score < best_score:
                    best_score = score
                    best_move = move
        
        # If all moves have same score, choose randomly
        if best_move is None and legal_moves:
            best_move = random.choice(legal_moves)
            
        return str(best_move)
    
    def minimax(self, game):
        """
        Recursive implementation of Minimax algorithm with caching
        
        Returns:
            Score from X player's perspective
        """
        # Create a hashable representation of the game state
        state_key = self._get_state_key(game)
        
        # Check if this state has already been evaluated
        if state_key in self.score_cache:
            return self.score_cache[state_key]
        
        winner = game.check_winner()
        if winner is not None:
            if winner == 1: 
                score = 10     # X wins
            elif winner == -1: 
                score = -10    # O wins
            else:
                score = 0      # Draw
            
            # Cache the result
            self.score_cache[state_key] = score
            return score
        
        current_player = game.get_current_player()
        if current_player == 1:  # X's turn, X maximizes score
            best_score = -float('inf')
            for move in game.get_legal_moves():
                temp_game = game.clone()
                temp_game.make_move(move)
                score = self.minimax(temp_game)
                best_score = max(score, best_score)
        else:  # O's turn, O minimizes X's score
            best_score = float('inf')
            for move in game.get_legal_moves():
                temp_game = game.clone()
                temp_game.make_move(move)
                score = self.minimax(temp_game)
                best_score = min(score, best_score)
        
        # Cache the result
        self.score_cache[state_key] = best_score
        return best_score
    
    def get_action_rewards(self, game) -> dict[str, float]:
        """
        Get reward values for every possible move from the current player's perspective.
        This can be used for training a reinforcement learning model.

        Args:
            game: Game object

        Returns:
            A dictionary mapping each legal move (as a string) to its minimax score.
        """
        player_value = game.get_current_player()
        legal_moves = game.get_legal_moves()
        rewards = {}

        if not legal_moves:
            return {}

        for move in legal_moves:
            temp_game = game.clone()
            temp_game.make_move(move)

            # The score is always from the perspective of player 1 (X).
            score = self.minimax(temp_game)

            # The reward for an action is the value of the resulting state from the current player's perspective.
            # If current player is X (1), reward is the score.
            # If current player is O (-1), reward is the negative of the score, as a good score for X is a bad state for O.
            reward = score if player_value == 1 else -score
            rewards[str(move)] = reward

        return rewards
    
    def _get_state_key(self, game):
        """
        Create a hashable representation of the TicTacToe game state
        
        Args:
            game: TicTacToe game object
            
        Returns:
            A tuple representing the board state and current player
        """
        # For TicTacToe, we can use a tuple of the board array and current player
        board_tuple = tuple(game.board)
        return (board_tuple, game.current_player)
    
    def supports_batch(self) -> bool:
        """Minimax agent supports batch processing, though it's serial"""
        return False
