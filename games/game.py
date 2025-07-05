from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Any, Dict

class Game(ABC):
    """Base abstract class for all games. All game implementations must inherit from this."""
    
    def __init__(self):
        """Initialize common game attributes"""
        self.current_player = 1  # Current player identifier
        self._game_over_forced_forfeit = False  # For forcing game end in evaluation
        self.system_prompt = None  # System prompt for LLM
        self.user_prompt_template = None  # User prompt template for LLM
        self.system_prompt_no_thinking = None  # System prompt without thinking requirement
        self.user_prompt_template_no_thinking = None  # User prompt template without thinking
        self.empty_symbol = '.'  # Symbol for empty positions
        self.name = self.__class__.__name__  # Default game name based on class name
    
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
    def get_chat_history_for_llm(self, llm) -> List[Dict[str, str]]:
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
    
    @abstractmethod
    def get_action_rewards(self) -> Dict[str, float]:
        """
        Get reward values for every possible move from the current player's perspective for reinforcement learning.
        Default implementation returns an empty dictionary.
        
        Subclasses should override this method to provide specific rewards.
        """
        return {}
    
    @abstractmethod
    def evaluate_position(self) -> float:
        """
        Evaluate the current position from player 1's perspective.
        Used by minimax when depth limit is reached.
        
        Returns:
            float: Position score from player 1's perspective.
                  Positive values favor player 1, negative favor player -1.
                  Should be in a reasonable range (e.g., -100 to 100).
        """
        pass
    
    @abstractmethod
    def clone(self):
        """Create a deep copy of the current game state"""
        pass