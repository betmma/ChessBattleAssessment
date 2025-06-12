from abc import ABC, abstractmethod
from typing import List, Any, Optional, Dict, Union

class Agent(ABC):
    """Abstract base class for all agents. All AI and rule-based agents inherit from this."""
    
    def __init__(self, name: str = "Agent"):
        self.name = name
    
    @abstractmethod
    def get_move(self, game, player_value: Any) -> Any:
        """
        Determine the next move from the game state
        
        Args:
            game: Game object
            player_value: Agent's player identifier in the game
            
        Returns:
            Next move, format depends on game implementation
        """
        pass
    
    @abstractmethod
    def supports_batch(self) -> bool:
        """Whether this agent supports batch processing multiple games"""
        pass
    
    def get_batch_moves(self, game_contexts: List[Dict]) -> List[Any]:
        """
        Get moves for multiple game states in batch, defaults to serial processing
        
        Args:
            game_contexts: List of game state and context dictionaries, each containing:
                           - 'game': Game object
                           - 'player_value': Agent's player identifier in that game
                           - Other possible context needed
        
        Returns:
            List of moves, corresponding to the input game contexts
        """
        if not self.supports_batch():
            return [self.get_move(ctx.get('game'), ctx.get('player_value')) for ctx in game_contexts]
        else:
            # Batch implementation to be overridden in subclasses
            raise NotImplementedError("Batch processing method should be implemented in subclasses that support batch")
            
    def __str__(self) -> str:
        return self.name