import os,sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from games.connect4 import Connect4Game
def connect4_reward_fn(data_source, solution_str, ground_truth, extra_info=None):
    choice=Connect4Game.parse_move_from_output(solution_str)
    if choice is None:
        return -1000
    return ground_truth.get(choice, -1000)