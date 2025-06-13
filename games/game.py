from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Any, Dict
from agents.agent import Agent

class Game(ABC):
    """Base abstract class for all games. All game implementations must inherit from this."""
    
    @abstractmethod
    def get_legal_moves(self) -> List[Any]:
        """Return all currently legal moves"""
        pass
    
    @abstractmethod
    def make_move(self, move: Any) -> bool:
        """Execute a move. Return True if move was legal and executed successfully, False otherwise"""
        pass
    
    @abstractmethod
    def check_winner(self) -> Optional[Any]:
        """Check if there's a winner. Return winner identifier, 0 for draw, None if game is not over"""
        pass
    
    @abstractmethod
    def is_game_over(self) -> bool:
        """Check if the game is over"""
        pass
    
    @abstractmethod
    def get_current_player(self) -> Any:
        """Get current player's turn"""
        pass
    
    @abstractmethod
    def get_state_representation(self) -> str:
        """Get a string representation of game state, for display or debugging"""
        pass
    
    @abstractmethod
    def get_chat_history_for_llm(self, llm: Agent) -> List[Dict[str, str]]:
        """Get the chat history to send to llm, including system prompt and game state"""
        pass
    
    @abstractmethod
    def parse_move_from_output(self, raw_output: str, legal_moves: List[Any]) -> Optional[Any]:
        """Parse a move from an agent's output string, validating against legal moves"""
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """Reset the game to initial state"""
        pass
        
    def force_forfeit(self) -> None:
        """
        Force the current player to forfeit the game.
        Default implementation does nothing, but subclasses should implement
        if they need special handling for forfeits.
        """
        pass