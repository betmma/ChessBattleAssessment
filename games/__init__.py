"""
Games package for ChessBattleAssessment
Contains game implementations and the base Game class.
"""

from .game import Game
from .tictactoe import TicTacToeGame
from .connect4 import Connect4Game
from .nim import NimGame # these 3 games are deprecated, use games in /boardGames instead
GamesList = []

def underscore_to_camel_case(name: str) -> str:
    """Convert snake_case to CamelCase."""
    return ''.join(word.capitalize() for word in name.split('_'))

# import all files under games/boardGames
import os
import pkgutil
# Dynamically import all game modules in the boardGames package and add to GamesList
board_games_path = os.path.dirname(__file__) + '/boardGames'
for _, module_name, _ in pkgutil.iter_modules([board_games_path]):
    module = __import__(f'games.boardGames.{module_name}', fromlist=[module_name])
    possibleNames = [module_name[0].upper()+module_name[1:], module_name.capitalize(), underscore_to_camel_case(module_name)]
    possibleNames.extend([name+'Game' for name in possibleNames])
    for name in possibleNames:
        if hasattr(module, name):
            GamesList.append(getattr(module, name))
            break
    else:
        raise ImportError(f"Module '{module_name}' does not define a game class with expected names: {possibleNames}")

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
    'Game',
    'Games',
    'GameByName',
]