"""
Evaluation package for ChessBattleAssessment
Contains evaluation tools and metrics for agent performance assessment.
"""

try:
    from .evaluator import Evaluator
except ImportError:
    Evaluator = None

__all__ = [
    'Evaluator'
]