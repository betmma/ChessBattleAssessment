"""
Games package for ChessBattleAssessment
Contains game implementations and the base Game class.
"""

from .game import Game
from .tictactoe import TicTacToeGame
from .connect4 import Connect4Game

__all__ = [
    'TicTacToeGame',
    'Connect4Game'
]