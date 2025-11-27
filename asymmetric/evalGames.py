from gameFilter import run_game_code_and_get_class
from dataclasses import dataclass

@dataclass
class TestSample:
    gameCode: str
    sequence: list
    finalBoard: list

# there will be 5 games and 100 test samples
testSamples:list[TestSample]=[]
testGamesCode:list[str]=[]

def batchAdd(gameCode:str, sequences:list[list[int]]):
    testGamesCode.append(gameCode)
    gameClass=run_game_code_and_get_class(gameCode)
    if gameClass is None:
        raise ValueError("Invalid game code. Test sample is corrupted.")
    for seq in sequences:
        game=gameClass()
        for move in seq:
            game.execute_move(move)
        testSamples.append(TestSample(gameCode=gameCode, sequence=seq, finalBoard=game.board))

game='''
from typing import List, Any, Optional, Tuple

class System(AbstractSystem):
    def create_initial_board(self) -> List[int]:
        return [0, 0]

    def get_legal_moves(self) -> List[int]:
        return [0, 1]

    def execute_move(self, move: int) -> None:
        if move == 0:
            self.board[0] += 1
        elif move == 1:
            self.board[0] += 1
            self.board[1] += self.board[0]
'''
sequences=[[0, 0], [0, 1], [1, 0], [1, 1], [0, 1], [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 0, 1, 1], [0, 1, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 1, 1], [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 1, 1], [0, 0, 0, 0, 0, 1, 0, 0]]
batchAdd(game, sequences)


game = '''
from typing import List, Any, Optional, Tuple

class System(AbstractSystem):
    def create_initial_board(self) -> List[int]:
        return [1, 0, 1, 0, 1]

    def get_legal_moves(self) -> List[int]:
        return list(range(len(self.board)))

    def execute_move(self, move: int) -> None:
        n=len(self.board)
        
        left_neighbor_idx = (move - 1 + n) % n
        right_neighbor_idx = (move + 1) % n
        
        self.board[left_neighbor_idx] = 1 - self.board[left_neighbor_idx]
        self.board[right_neighbor_idx] = 1 - self.board[right_neighbor_idx]
'''
sequences=[[0], [1], [2], [3], [4], [2, 4], [0, 1], [0, 2], [0, 3], [0, 4], [0, 1, 2], [0, 2, 3], [0, 2, 4], [1, 2, 4], [1, 3, 4], [0, 1, 2, 3], [0, 1, 2, 4], [0, 1, 3, 4], [0, 2, 3, 4], [1, 2, 3, 4]]
batchAdd(game, sequences)

game='''
from typing import List, Any, Optional, Tuple

class System(AbstractSystem):
    def create_initial_board(self) -> List[int]:
        return [10]

    def get_legal_moves(self) -> List[int]:
        return list(range(3))

    def execute_move(self, move: int) -> None:
        if move == 0:
            self.board[0] += 5
        elif move == 1:
            self.board[0] = int(str(self.board[0])[::-1])
        elif move == 2:
            self.board[0] -= 3
            self.board[0] = max(self.board[0], 0)
'''
sequences=[[0, 0], [0, 2], [0, 1], [1, 2], [1, 0], [0, 0, 2, 1], [0, 0, 2, 0], [0, 0, 2, 2], [0, 0, 1, 2], [0, 0, 0, 1], [0, 0, 2, 1, 0, 2], [0, 0, 2, 1, 0, 1], [0, 0, 2, 1, 2, 1], [0, 0, 2, 0, 0, 1], [0, 0, 2, 0, 0, 2], [0, 0, 2, 1, 0, 2, 1, 2], [0, 0, 2, 0, 0, 2, 0, 1], [0, 0, 2, 0, 2, 1, 0, 0], [0, 0, 1, 2, 2, 2, 2, 2], [0, 0, 0, 1, 2, 1, 0, 0]]
batchAdd(game, sequences)

game='''
from typing import List, Any, Optional, Tuple

class System(AbstractSystem):
    def create_initial_board(self) -> List[List[int]]:
        # 3 stacks
        return [[3, 2, 1], [], []]

    def get_legal_moves(self) -> List[Tuple[int, int]]:
        # Move top item from stack i to stack j
        moves = []
        for i in range(3):
            for j in range(3):
                if i != j and self.board[i] and (not self.board[j] or self.board[j][-1] > self.board[i][-1]):
                    # Legal if dest is empty OR top of dest > moving item (Towers of Hanoi rules)
                    moves.append((i, j))
        return moves

    def execute_move(self, move: Tuple[int, int]) -> None:
        src, dst = move
        val = self.board[src].pop()
        self.board[dst].append(val)
'''
sequences=[[(0, 1), (0, 2)], [(0, 2), (0, 1)], [(0, 1), (0, 2), (1, 2)], [(0, 1), (0, 2), (1, 0)], [(0, 2), (0, 1), (2, 1)], [(0, 2), (0, 1), (2, 0)], [(0, 1), (0, 2), (1, 2), (0, 1)], [(0, 2), (0, 1), (2, 1), (0, 2)], [(0, 1), (0, 2), (1, 2), (0, 1), (2, 0)], [(0, 1), (0, 2), (1, 2), (0, 1), (2, 1)], [(0, 2), (0, 1), (2, 1), (0, 2), (1, 0)], [(0, 2), (0, 1), (2, 1), (0, 2), (1, 2)], [(0, 1), (0, 2), (1, 2), (0, 1), (2, 0), (2, 1)], [(0, 1), (0, 2), (1, 2), (0, 1), (2, 1), (2, 0)], [(0, 2), (0, 1), (2, 1), (0, 2), (1, 0), (1, 2)], [(0, 2), (0, 1), (2, 1), (0, 2), (1, 2), (1, 0)], [(0, 1), (0, 2), (1, 2), (0, 1), (2, 0), (2, 1), (0, 2)], [(0, 1), (0, 2), (1, 2), (0, 1), (2, 0), (2, 1), (0, 1)], [(0, 1), (0, 2), (1, 2), (0, 1), (2, 1), (2, 0), (1, 0)], [(0, 1), (0, 2), (1, 2), (0, 1), (2, 1), (2, 0), (1, 2)]]
batchAdd(game, sequences)

game='''
from typing import List, Any, Optional, Tuple
class System(AbstractSystem):
    def create_initial_board(self) -> List[int]:
        return [1, 2, 3, 4, 5]

    def get_legal_moves(self) -> List[int]:
        # Choose index k (1 to length-1). 
        # Flip the sub-segment board[0...k]
        return list(range(1, len(self.board)))

    def execute_move(self, k: int) -> None:
        # Reverse the first k+1 elements
        # We act on slice [0 : k+1]
        subset = self.board[:k+1]
        self.board[:k+1] = subset[::-1]
'''
sequences = [[1, 4], [1, 3], [1, 2], [3, 4], [3, 1], [1, 4, 3], [1, 4, 2], [1, 4, 1], [1, 3, 1], [1, 3, 4], [1, 4, 3, 2], [1, 4, 3, 1], [1, 4, 2, 4], [1, 4, 2, 1], [1, 4, 1, 4], [1, 4, 3, 1, 2], [1, 4, 2, 1, 3], [1, 4, 1, 3, 1], [1, 4, 1, 3, 2], [1, 3, 1, 4, 1]]
batchAdd(game, sequences)


