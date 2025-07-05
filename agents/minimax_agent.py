import sys
import os,random

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from agents.agent import Agent
from agents.universal_minimax_agent import UniversalMinimaxAgent
from agents.random_agent import RandomAgent

class MinimaxAgent(Agent):
    """
    Universal Minimax agent that routes to specific game implementations
    """
    
    def __init__(self, name: str = "Default", depth: int = 4, random_chance: float = 0.0):
        super().__init__(name)
        self.depth = depth
        self.universal_agent = UniversalMinimaxAgent(name=f"{name}-Universal", max_depth=depth)
        self.random_chance = random_chance
        self.random_agent = RandomAgent()
        if self.name=="Default":
            self.name = f"Minimax-random-{random_chance}-depth-{depth}"
    
    def get_move(self, game) -> str:
        """
        Get a move using appropriate minimax implementation based on game type, with optional random chance
        
        Args:
            game: Game object
            
        Returns:
            str: The chosen move as a string
        """
        
        if random.random() < self.random_chance:
            # Use random agent with specified chance
            return self.random_agent.get_move(game)
        else:
            return self.universal_agent.get_move(game)

    def set_depth(self, depth: int):
        """
        Set the search depth for both agents
        
        Args:
            depth: Maximum search depth
        """
        self.depth = depth
        self.connect4_agent.max_depth = depth
        # TicTacToe doesn't use depth limit as it can solve completely

    def get_action_rewards(self, game) -> dict[str, float]:
        """
        Get action rewards for the current game state
        
        Args:
            game: Game object
            
        Returns:
            dict[str, float]: Dictionary of action rewards
        """
        return self.universal_agent.get_action_rewards(game)