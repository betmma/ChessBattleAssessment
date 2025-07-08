"""
Games package for ChessBattleAssessment
Contains game implementations and the base Game class.
"""

from .game import Game
from .tictactoe import TicTacToeGame
from .connect4 import Connect4Game
from .nim import NimGame
from .boardGames.capture import Capture
GamesList = [Connect4Game, TicTacToeGame,  NimGame, Capture]
Games = {game.__name__.removesuffix('Game'): game for game in GamesList}
def GameByName(name: str) -> Game:
    """
    Get a game class by its name.
    
    Args:
        name (str): Name of the game (e.g., 'TicTacToe', 'Connect4', 'Nim').
    
    Returns:
        Game: The corresponding game class.
    
    Raises:
        ValueError: If the game name is not recognized.
    """
    if name in Games:
        return Games[name]
    else:
        raise ValueError(f"Game '{name}' is not recognized. All available games: {list(Games.keys())}")

__all__ = [
    'TicTacToeGame',
    'Connect4Game',
    'NimGame',
    'Capture',
    'Game',
    'Games',
    'GameByName',
]