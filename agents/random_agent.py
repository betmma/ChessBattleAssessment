import random
from typing import List, Dict
from .agent import Agent

class RandomAgent(Agent):
    """Agent that makes random legal moves"""
    
    def __init__(self, name: str = "RandomAgent", seed: int = None):
        super().__init__(name)
        if seed is not None:
            random.seed(seed)
    
    def get_move(self, game) -> str:
        """
        Get a random move from the legal moves
        
        Args:
            game: Game object
            
        Returns:
            str: Random legal move as string
        """
        legal_moves = game.get_legal_moves()
        if not legal_moves:
            return 'No legal moves available'
        
        move = random.choice(legal_moves)
        return str(move)