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
    NO_CODE_FOUND=-1.0
    CANT_EXTRACT_CLASS=-1.0
    ERROR=-1.0
    TIMEOUT=-1.0
    MOVES_NOT_STRINGIFIABLE=-1.0
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
    for move in legal_moves1: # some uses function that is not stringifiable
        if type(move)==str:
            continue # string should pass, while below eval(str(move)) causes error (interpreting as a variable)
        try:
            response=eval(str(move))
        except Exception as e:
            return FilterGameResult.MOVES_NOT_STRINGIFIABLE
        if response!=move:
            return FilterGameResult.MOVES_NOT_STRINGIFIABLE
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

# Worker function for multiprocessing timeout handling
from multiprocessing import Process, Queue

def _filterGame_worker(GameClass, q: Queue):
    ret = _filterGame(GameClass)
    q.put(ret.name)

import re
def extract_game(text: str) -> Optional[str]:
    code_blocks= re.findall(r"```(?:python)?\n(.*?)\n```", text, re.DOTALL | re.IGNORECASE)
    if code_blocks:
        return code_blocks[-1].strip()
    return text.split('</think>')[-1].strip()

def run_game_code_and_get_class(game_src: str):
    """
    Exec the generated Game class in a minimal namespace that already defines AbstractSystem.
    Returns Game class object. If not found or doesn't pass check, return None
    """
    namespace = {'AbstractSystem': AbstractSystem}
    try:
        exec(game_src, namespace, namespace)
    except Exception as e:
        print(f'Error executing game code: {e}')
        return None
    Game = namespace.get("System", None)
    if Game is None:
        print("Generated code did not define class System.")
        return None
    return Game


LOG_PATH="logs/gameFilter.log"
import os, inspect, traceback

# Ensure log directory exists once
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

resultCounter={}
count=0

def filterGame(response:str)->FilterGameResult:
    global count
    count+=1
    ret=None
    gameCode=extract_game(response)
    if gameCode is None:
        ret=FilterGameResult.NO_CODE_FOUND
    else:
        GameClass=run_game_code_and_get_class(gameCode)
        if GameClass is None:
            ret=FilterGameResult.CANT_EXTRACT_CLASS
    if ret is None:
        try:
            q = Queue()
            p = Process(target=_filterGame_worker, args=(GameClass, q))
            p.start()
            p.join(5)
            if p.is_alive():
                p.terminate()
                p.join()
                ret=FilterGameResult.TIMEOUT
            else:
                name = q.get_nowait()
                ret = FilterGameResult[name]
        except Exception as e:
            stack=traceback.format_exc()
            print(f"Error occurred while filtering game {GameClass}:\n{stack}")
            with open(LOG_PATH,"a") as f:
                f.write(f"ID:{count}, error stack:\n{stack}\n")
            ret=FilterGameResult.ERROR
    code=gameCode if gameCode is not None else "<no code>"
    if len(code)>10000: # probably repeating something so not useful to log all
        code=code.replace('\n','\\n') # reduce lines
        code=code[:500]+"\n...\n"+code[-500:]
    resultCounter[ret.name]=resultCounter.get(ret.name,0)+1
    with open(LOG_PATH,"a") as f:
        f.write(f"ID:{count}, Code:\n```\n{code}\n```\nResult: {ret.name}\n")
        f.write(f"Summary so far: {resultCounter}\n\n")
    return ret