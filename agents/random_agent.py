import random
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from agents.agent import Agent

class RandomAgent(Agent):
    """Agent that randomly selects from legal moves"""
    
    def __init__(self, name: str = "RandomAgent"):
        super().__init__(name)
    
    def get_move(self, game, player_value) -> any:
        """Randomly select from legal moves"""
        legal_moves = game.get_legal_moves()
        if not legal_moves:
            return None
        return random.choice(legal_moves)
    
    def supports_batch(self) -> bool:
        """Random agent supports batch processing"""
        return True
    
    def get_batch_moves(self, game_contexts):
        """Process multiple games in batch"""
        moves = []
        for context in game_contexts:
            game = context.get('game')
            player_value = context.get('player_value')
            moves.append(self.get_move(game, player_value))
        return moves