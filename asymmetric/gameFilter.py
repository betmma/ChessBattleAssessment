import random, aenum
from abc import abstractmethod
from typing import List, Any, Optional, Tuple

class AbstractSystem():
    __slots__ = ['board'] # Can't add other attributes
    def __init__(self):
        super().__init__()
        self.board = self.create_initial_board()

    # ----------------------------------------------------------------
    # Abstract methods to be implemented by specific one player game classes
    # ----------------------------------------------------------------

    @abstractmethod
    def create_initial_board(self) -> list:
        """
        Creates and returns the initial board configuration.
        """
        pass
    
    @abstractmethod
    def get_legal_moves(self) -> List[Any]:
        """Return all currently legal moves."""
        pass

    @abstractmethod
    def execute_move(self, move: Any) -> None:
        """Executes a move. self.board will be changed."""
        pass

class FilterGameResult(aenum.Enum):
    _settings_ = aenum.NoAlias
    PASS=0.0
    ERROR=-1.0
    POSSIBLE_STATES_IS_1=-1.0
    NOT_DEFINITE=-1.0
    NO_LEGAL_MOVE_IN_3_MOVES=-0.5
    POSSIBLE_STATES_LESS_THAN_5=-0.5
    POSSIBLE_STATES_LESS_THAN_20=-0.3

from copy import deepcopy
def _filterGame(GameClass: AbstractSystem)->FilterGameResult:
    game1=GameClass()
    legal_moves1=game1.get_legal_moves()
    if len(legal_moves1)==0:
        return FilterGameResult.POSSIBLE_STATES_IS_1
    g = GameClass()
    g2 = GameClass()
    for i in range(3):
        if len(g.get_legal_moves())==0:
            return FilterGameResult.NO_LEGAL_MOVE_IN_3_MOVES
        g.execute_move(g.get_legal_moves()[0])
        if len(g2.get_legal_moves())==0:
            return FilterGameResult.NO_LEGAL_MOVE_IN_3_MOVES
        g2.execute_move(g2.get_legal_moves()[0])
    if g.board!=g2.board:
        return FilterGameResult.NOT_DEFINITE
    states=[game1.board]
    count=0
    queue=[game1]
    while len(queue)>count and count<100:
        game=queue[count]
        count+=1
        legal_moves=game.get_legal_moves()
        for legal_move in legal_moves:
            gamei=GameClass()
            gamei.board=deepcopy(game.board)
            gamei.execute_move(legal_move)
            if gamei.board not in states:
                states.append(gamei.board)
                queue.append(gamei)
    if len(states)==1:
        return FilterGameResult.POSSIBLE_STATES_IS_1
    if len(states)<5:
        return FilterGameResult.POSSIBLE_STATES_LESS_THAN_5
    if len(states)<20:
        return FilterGameResult.POSSIBLE_STATES_LESS_THAN_20
    return FilterGameResult.PASS

LOG_PATH="logs/gameFilter.log"
import inspect,traceback
resultCounter={}
count=0
def filterGame(GameClass: AbstractSystem, gameCode:str)->FilterGameResult:
    global count
    count+=1
    ret=None
    try:
        ret=_filterGame(GameClass)
    except Exception as e:
        stack=traceback.format_exc()
        print(f"Error occurred while filtering game {GameClass}:\n{stack}")
        with open(LOG_PATH,"a") as f:
            f.write(f"ID:{count}, error stack:\n{stack}\n")
        ret=FilterGameResult.ERROR
    code=gameCode
    resultCounter[ret.name]=resultCounter.get(ret.name,0)+1
    with open(LOG_PATH,"a") as f:
        f.write(f"ID:{count}, Code:\n```\n{code}\n```\nResult: {ret.name}\n")
        f.write(f"Summary so far: {resultCounter}\n\n")
    return ret