from abc import ABC, abstractmethod
from typing import List, Dict, Any

class Agent(ABC):
    """Base abstract class for all agents."""
    
    def __init__(self, name: str = "BaseAgent"):
        self.name = name
    
    @abstractmethod
    def get_move(self, game) -> str:
        """
        Get a move from the agent
        
        Args:
            game: Game object
            
        Returns:
            str: The chosen move as a string
        """
        pass
    
    def get_batch_moves(self, game_contexts: List[Dict]) -> List[str]:
        """
        Get moves for multiple games in batch
        
        Args:
            game_contexts: List of dictionaries with:
                           - 'game': Game object
                           
        Returns:
            List[str]: List of moves as strings
        """
        moves = []
        for context in game_contexts:
            game = context.get('game')
            move = self.get_move(game)
            moves.append(move)
        return moves
    
    def __str__(self) -> str:
        return self.name