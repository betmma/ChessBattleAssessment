"""
Agents package for ChessBattleAssessment
Contains various AI agent implementations for game playing.
"""

from .agent import Agent
from .random_agent import RandomAgent
from .minimax_agent import MinimaxAgent
from .minimax_agent_tictactoe import MinimaxAgentTicTacToe
from .minimax_agent_connect4 import MinimaxAgentConnect4

# Import API and VLLM agents if dependencies are available
try:
    from .api_agent import APIAgent
except ImportError:
    APIAgent = None

try:
    from .vllm_agent import VLLMAgent
except ImportError:
    VLLMAgent = None

__all__ = [
    'Agent',
    'RandomAgent', 
    'MinimaxAgent',
    'MinimaxAgentTicTacToe',
    'MinimaxAgentConnect4',
    'APIAgent',
    'VLLMAgent'
]