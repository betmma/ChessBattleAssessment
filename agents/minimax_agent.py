import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from agents.agent import Agent
from agents.minimax_agent_tictactoe import MinimaxAgentTicTacToe
from agents.minimax_agent_connect4 import MinimaxAgentConnect4

class MinimaxAgent(Agent):
    """
    Universal Minimax agent that routes to specific game implementations
    """
    
    def __init__(self, name: str = "MinimaxAgent", depth: int = 4):
        super().__init__(name)
        self.depth = depth
        self.tictactoe_agent = MinimaxAgentTicTacToe(f"{name}-TicTacToe")
        self.connect4_agent = MinimaxAgentConnect4(f"{name}-Connect4", max_depth=depth)
    
    def get_move(self, game) -> str:
        """
        Get a move using appropriate minimax implementation based on game type
        
        Args:
            game: Game object
            
        Returns:
            str: The chosen move as a string
        """
        # Lazy import to avoid circular import
        from games import TicTacToeGame, Connect4Game
        
        if isinstance(game, TicTacToeGame):
            return self.tictactoe_agent.get_move(game)
        elif isinstance(game, Connect4Game):
            return self.connect4_agent.get_move(game)
        else:
            raise ValueError(f"Unsupported game type: {type(game)}")
    
    def set_depth(self, depth: int):
        """
        Set the search depth for both agents
        
        Args:
            depth: Maximum search depth
        """
        self.depth = depth
        self.connect4_agent.max_depth = depth
        # TicTacToe doesn't use depth limit as it can solve completely
    
    def get_supported_games(self):
        """
        Get list of supported game types
        
        Returns:
            List[str]: List of supported game types
        """
        return ['tictactoe', 'connect4']
    
    def supports_batch(self) -> bool:
        """Minimax agent supports batch processing, though it's serial"""
        return False