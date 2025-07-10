from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Any, Dict

class Game(ABC):
    """Base abstract class for all games. All game implementations must inherit from this."""
    
    system_prompt = None  # System prompt for LLM
    # User prompt template for LLM. example:
    #    ("{board_representation}\n"
    #     "You are player '{player_symbol}'.\n"
    #     "Your available legal moves (columns): [{legal_moves_str}]\n"
    #     "Provide your thinking and final move in the specified format: `(column_number)`")
    user_prompt_template = None
    system_prompt_no_thinking = None  # System prompt without thinking requirement
    user_prompt_template_no_thinking = None  # User prompt template without thinking
    
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.name = cls.__name__.removesuffix('Game')
        
    def __init__(self):
        """Initialize common game attributes"""
        self.current_player = 1  # Current player identifier
        self._game_over_forced_forfeit = False  # For forcing game end in evaluation
        self.empty_symbol = '.'  # Symbol for empty positions
        self.name = self.__class__.name  # Default game name based on class name
    
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
    def get_board_representation_for_llm(self) -> str:
        """Get board state representation for LLM"""
        pass
    
    @abstractmethod
    def get_key_for_cache(self) -> tuple:
        """Get a unique key for caching game state, without current player or game over state"""
        pass
    
    def get_state_representation(self) -> str:
        """Get a string representation of game state, for display or debugging"""
        # Use the LLM representation as base and add metadata
        from utils import clean_np_types
        board_repr = self.get_board_representation_for_llm()
        current_player_symbol = self._get_player_symbol(self.current_player)
        legal_moves = clean_np_types(self.get_legal_moves())
        
        state_str = board_repr + "\n"
        state_str += f"Current turn: {current_player_symbol} (plays as {self.current_player})\n"
        state_str += f"Legal moves: {legal_moves}\n"
        
        return state_str
    
    @abstractmethod
    def load_state_from_representation(self, state_str: str) -> bool:
        """
        Load game state from a string representation (reverse of get_state_representation).
        
        Args:
            state_str: String representation of game state, typically from get_state_representation()
            
        Returns:
            bool: True if state was loaded successfully, False if parsing failed
        """
        pass
    
    @abstractmethod
    def _get_player_symbol(self, player_value: Any) -> str:
        """Get the symbol representation for a player"""
        pass
    
    def get_chat_history_for_llm(self, llm) -> List[Dict[str, str]]:
        """Get the chat history to send to llm, including system prompt and game state"""
        from agents.vllm_agent import VLLMAgent
        
        board_representation = self.get_board_representation_for_llm()
        player_symbol = self._get_player_symbol(self.current_player)
        legal_moves = self.get_legal_moves()
        legal_moves_str = self._format_legal_moves_for_prompt(legal_moves)
        
        system_prompt = self.system_prompt
        user_prompt_template = self.user_prompt_template
        
        if isinstance(llm, VLLMAgent) and not llm.enable_thinking:
            system_prompt = self.system_prompt_no_thinking
            user_prompt_template = self.user_prompt_template_no_thinking
            
        user_prompt = user_prompt_template.format(
            board_representation=board_representation,
            player_symbol=player_symbol,
            legal_moves_str=legal_moves_str
        )
        
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    
    def _format_legal_moves_for_prompt(self, legal_moves: List[Any]) -> str:
        """Format legal moves for display in prompt. Can be overridden by subclasses."""
        return ", ".join([str(move) for move in legal_moves])
    
    @abstractmethod
    def parse_move_from_output(self, raw_output: str) -> Optional[Any]:
        """Parse a move from an agent's output string"""
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
    
    def get_action_rewards(self) -> Dict[str, float]:
        """
        Get reward values for every possible move from the current player's perspective for reinforcement learning.
        Default implementation uses minimax agent.
        
        Subclasses should override this method if minimax is not suitable or if they have a custom implementation.
        """
        from agents.minimax_agent import MinimaxAgent
        if not hasattr(self.__class__, '_minimax_agent'): # each class has an independent minimax agent, instead of each game instance
            self.__class__._minimax_agent = MinimaxAgent()
        return self.__class__._minimax_agent.get_action_rewards(self)

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