from typing import List, Tuple, Optional, Any
import random
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents import Agent

class MinimaxAgentConnect4(Agent):
    """Agent using Minimax algorithm with alpha-beta pruning"""
    
    def __init__(self, name: str = "MinimaxAgent", max_depth: int = 4):
        super().__init__(name)
        # Cache for storing evaluated positions
        self.score_cache = {}
        # Maximum search depth for Connect4
        self.max_depth = max_depth
    
    def get_move(self, game) -> str:
        """
        Get best move using Minimax algorithm with alpha-beta pruning
        
        Args:
            game: Game object
            
        Returns:
            str: Best move as string
        """
        player_value = game.get_current_player()
        best_move = None
        legal_moves = game.get_legal_moves()

        if not legal_moves: 
            return 'No legal moves available'

        # Clear cache for each new move decision to prevent memory growth
        self.score_cache = {}

        if player_value == 1:  # X (maximizing player)
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
        else:  # O (minimizing player)
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
    
    def minimax(self, game):
        """
        Legacy minimax method (kept for compatibility)
        """
        return self.minimax_alpha_beta(game, -float('inf'), float('inf'), game.get_current_player() == 1, 0)
    
    def minimax_alpha_beta(self, game, alpha, beta, maximizing_player, depth):
        """
        Minimax algorithm with alpha-beta pruning and depth limit
        
        Args:
            game: Game object
            alpha: Best value that maximizing player can guarantee so far
            beta: Best value that minimizing player can guarantee so far
            maximizing_player: True if current player is maximizing (X), False if minimizing (O)
            depth: Current search depth
            
        Returns:
            Score from X player's perspective
        """
        # Create a hashable representation of the game state
        state_key = self._get_state_key(game, depth)
        
        # Check if this state has already been evaluated
        if state_key in self.score_cache:
            return self.score_cache[state_key]
        
        winner = game.check_winner()
        if winner is not None:
            if winner == 1: 
                score = 1000 - depth     # X wins (prefer shorter paths to victory)
            elif winner == -1: 
                score = -1000 + depth    # O wins (prefer longer paths to defeat)
            else:
                score = 0                # Draw
            
            # Cache the result
            self.score_cache[state_key] = score
            return score
        
        # If we've reached maximum depth, use heuristic evaluation
        if depth >= self.max_depth:
            score = self._evaluate_position(game)
            self.score_cache[state_key] = score
            return score
        
        if maximizing_player:  # X's turn, maximizing player
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
            self.score_cache[state_key] = max_eval
            return max_eval
        else:  # O's turn, minimizing player
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
            self.score_cache[state_key] = min_eval
            return min_eval
    
    def _evaluate_position(self, game):
        """
        Heuristic evaluation function for Connect4 positions
        
        Args:
            game: Game object
            
        Returns:
            Score from X player's perspective
        """
        board = game.board
        rows, cols = len(board), len(board[0])
        score = 0
        
        # Evaluate all possible 4-in-a-row positions
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]  # horizontal, vertical, diagonal
        
        for row in range(rows):
            for col in range(cols):
                for dr, dc in directions:
                    window = []
                    for i in range(4):
                        r, c = row + i * dr, col + i * dc
                        if 0 <= r < rows and 0 <= c < cols:
                            window.append(board[r][c])
                        else:
                            break
                    
                    if len(window) == 4:
                        score += self._evaluate_window(window)
        
        return score
    
    def _evaluate_window(self, window):
        """
        Evaluate a 4-piece window for Connect4
        
        Args:
            window: List of 4 pieces (1 for X, -1 for O, 0 for empty)
            
        Returns:
            Score for this window
        """
        score = 0
        x_count = window.count(1)
        o_count = window.count(-1)
        empty_count = window.count(0)
        
        # If both players have pieces in the window, it's blocked
        if x_count > 0 and o_count > 0:
            return 0
        
        # Score for X (maximizing player)
        if x_count == 4:
            score += 100
        elif x_count == 3 and empty_count == 1:
            score += 10
        elif x_count == 2 and empty_count == 2:
            score += 2
        
        # Score for O (minimizing player)
        if o_count == 4:
            score -= 100
        elif o_count == 3 and empty_count == 1:
            score -= 10
        elif o_count == 2 and empty_count == 2:
            score -= 2
        
        return score
    
    def _get_state_key(self, game, depth=0):
        """
        Create a hashable representation of the game state
        
        Args:
            game: Game object
            depth: Current search depth
            
        Returns:
            A tuple representing the board state, current player, and depth
        """
        board = game.board
        board = [cell for row in board for cell in row]
        board_tuple = tuple(board)
        return (board_tuple, game.current_player, depth)
    
    def supports_batch(self) -> bool:
        """Minimax agent supports batch processing, though it's serial"""
        return False
