"""
Utils package for ChessBattleAssessment
Contains utility functions and helper classes.
"""

try:
    from .model_utils import ModelUtils
except ImportError:
    ModelUtils = None

__all__ = [
    'ModelUtils'
]