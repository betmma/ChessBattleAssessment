"""
Utils package for ChessBattleAssessment
Contains utility functions and helper classes.
"""

try:
    from .model_utils import ModelUtils
except ImportError:
    ModelUtils = None

from .create_agent import create_agent

__all__ = [
    'ModelUtils'
    'create_agent'
]