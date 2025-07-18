import sys
import os,random
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from agents.agent import Agent
from agents.universal_minimax_agent import UniversalMinimaxAgent
from agents.random_agent import RandomAgent

class MinimaxAgent(Agent):
    """
    Universal Minimax agent that routes to specific game implementations
    """
    
    def __init__(self, name: str = "Default", depth: int = 4, random_chance: float = 0.0, temperature: float = 0.0):
        super().__init__(name)
        self.depth = depth
        self.universal_agent = UniversalMinimaxAgent(name=f"{name}-Universal", max_depth=depth)
        self.random_chance = random_chance
        self.temperature = temperature
        self.random_agent = RandomAgent()
        if self.name=="Default":
            temp_suffix = f"-temp-{temperature}" if temperature > 0 else ""
            self.name = f"Minimax-random-{random_chance}-depth-{depth}{temp_suffix}"
    
    def get_move(self, game) -> str:
        """
        Get a move using appropriate minimax implementation based on game type, with optional random chance or temperature
        
        Args:
            game: Game object
            
        Returns:
            str: The chosen move as a string
        """
        
        # Old random implementation - use random agent with specified chance
        if self.random_chance > 0 and random.random() < self.random_chance:
            return self.random_agent.get_move(game)
        
        # Temperature-based selection
        if self.temperature > 0:
            return self._temperature_based_move(game)
        
        # Default deterministic minimax
        return self.universal_agent.get_move(game)
    
    def _temperature_based_move(self, game) -> str:
        """
        Select a move using temperature-based probability distribution over action rewards
        
        Args:
            game: Game object
            
        Returns:
            str: The chosen move as a string
        """
        action_rewards = self.universal_agent.get_action_rewards(game)
        
        if not action_rewards:
            # Fallback to random if no rewards available
            return self.random_agent.get_move(game)
        
        moves = list(action_rewards.keys())
        rewards = list(action_rewards.values())
        
        # Apply temperature scaling to rewards
        # Higher temperature = more random, lower temperature = more deterministic
        scaled_rewards = np.array(rewards) / self.temperature if self.temperature > 0 else np.array(rewards) * 1000
        
        # Apply softmax to get probabilities
        # Subtract max for numerical stability
        exp_rewards = np.exp(scaled_rewards - np.max(scaled_rewards))
        probabilities = exp_rewards / np.sum(exp_rewards)
        
        # Sample move based on probabilities
        chosen_move = np.random.choice(moves, p=probabilities)
        return chosen_move

    def set_depth(self, depth: int):
        """
        Set the search depth for the universal agent
        
        Args:
            depth: Maximum search depth
        """
        self.depth = depth
        self.universal_agent.max_depth = depth

    def get_action_rewards(self, game) -> dict[str, float]:
        """
        Get action rewards for the current game state
        
        Args:
            game: Game object
            
        Returns:
            dict[str, float]: Dictionary of action rewards
        """
        return self.universal_agent.get_action_rewards(game)