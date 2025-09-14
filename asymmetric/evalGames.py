from gameFilter import run_game_code_and_get_class
from dataclasses import dataclass

@dataclass
class TestSample:
    gameCode: str
    sequence: list
    finalBoard: list

# there will be 5 games and 100 test samples
testSamples:list[TestSample]=[]

def batchAdd(gameCode:str, sequences:list[list[int]]):
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
        return [0, 0, 0]

    def get_legal_moves(self) -> List[int]:
        return [0, 1, 2]

    def execute_move(self, move: int) -> None:
        self.board[0] += 1
        if move >= 1:
            self.board[1] += self.board[0]
        if move >= 2:
            self.board[2] += self.board[1]
'''
sequences=[[2, 0], [2, 1], [0, 2], [1, 0], [1, 1], [2, 0, 2, 0], [1, 1, 2, 1], [2, 1, 0, 2], [2, 0, 1, 0], [0, 2, 1, 1], [2, 2, 1, 1, 2, 1], [2, 0, 1, 1, 1, 2], [0, 2, 2, 0, 1, 2], [0, 1, 2, 2, 1, 1], [2, 2, 2, 0, 1, 0], [0, 1, 2, 1, 1, 1, 2, 0], [1, 1, 0, 0, 1, 1, 0, 1], [2, 1, 2, 2, 0, 2, 2, 2], [0, 2, 1, 2, 2, 1, 0, 1], [0, 1, 0, 2, 1, 1, 2, 0]]
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
        self.board[move] += 1
        
        left_neighbor_idx = (move - 1 + n) % n
        right_neighbor_idx = (move + 1) % n
        
        self.board[left_neighbor_idx] = 1 - self.board[left_neighbor_idx]
        self.board[right_neighbor_idx] = 1 - self.board[right_neighbor_idx]
'''
sequences=[[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 2], [0, 0, 0, 3], [0, 0, 0, 4], [3, 4, 1, 2, 2, 2], [1, 4, 2, 4, 1, 3], [0, 2, 2, 2, 0, 4], [2, 0, 3, 0, 0, 1], [1, 2, 1, 3, 4, 3], [0, 1, 3, 1, 4, 2, 1, 0], [1, 0, 1, 3, 4, 0, 3, 2], [3, 0, 0, 3, 3, 2, 4, 1], [2, 1, 0, 3, 1, 1, 0, 2], [4, 1, 2, 1, 3, 4, 0, 4]]
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
            self.board[0] += self.board[0] & (self.board[0] + 31)
        elif move == 1:
            self.board[0] = int(str(self.board[0])[::-1])
        elif move == 2:
            self.board[0] |= self.board[0] + 1
'''
sequences=[[0, 0], [0, 1], [0, 2], [1, 2], [2, 0], [0, 0, 1, 0], [0, 0, 1, 2], [0, 0, 2, 2], [0, 1, 0, 0], [0, 1, 0, 2], [0, 0, 1, 2, 1, 0], [0, 0, 2, 2, 1, 0], [0, 1, 0, 0, 0, 0], [0, 1, 0, 0, 0, 2], [0, 1, 0, 0, 1, 0], [1, 0, 0, 0, 0, 0, 1, 2], [1, 0, 0, 1, 0, 0, 1, 1], [1, 0, 1, 0, 2, 2, 1, 0], [0, 2, 0, 1, 2, 1, 2, 0], [0, 0, 1, 2, 0, 0, 0, 2]]
batchAdd(game, sequences)

game='''
from typing import List, Any, Optional, Tuple

class System(AbstractSystem):
    def create_initial_board(self) -> List[int]:
        return [2, 3, 5, 7, 11, 13]

    def get_legal_moves(self) -> List[Tuple[int, int]]:
        moves = []
        n = len(self.board)
        for i in range(n):
            for j in range(n):
                if i != j:
                    moves.append((i, j))
        return moves

    def execute_move(self, move: Tuple[int, int]) -> None:
        i, j = move
        val_i, val_j = self.board[i], self.board[j]
        if val_i != 0:
            self.board[j] = val_j // val_i
            self.board[i] = val_i + 1
        self.board[i], self.board[j] = self.board[j], self.board[i]
'''
sequences=[[(0, 1), (0, 1)], [(0, 1), (0, 2)], [(0, 1), (0, 3)], [(0, 1), (0, 4)], [(0, 1), (0, 5)], [(3, 1), (1, 0), (2, 4), (4, 1)], [(0, 4), (2, 0), (3, 0), (0, 5)], [(1, 4), (1, 0), (5, 3), (5, 3)], [(3, 5), (1, 5), (0, 5), (5, 1)], [(3, 5), (3, 1), (4, 5), (1, 4)], [(1, 4), (0, 5), (2, 0), (5, 1), (2, 4), (0, 3)], [(1, 5), (3, 4), (3, 5), (2, 3), (1, 0), (0, 1)], [(0, 5), (2, 4), (1, 2), (4, 5), (2, 0), (4, 0)], [(5, 4), (1, 2), (4, 3), (5, 3), (0, 5), (2, 1)], [(1, 3), (4, 5), (2, 5), (3, 4), (0, 3), (5, 4)], [(5, 3), (1, 4), (0, 1), (3, 0), (1, 0), (2, 4), (1, 5), (3, 4)], [(2, 5), (5, 4), (2, 1), (5, 2), (1, 2), (2, 3), (4, 5), (0, 4)], [(3, 1), (1, 2), (0, 4), (5, 1), (4, 1), (5, 4), (3, 4), (3, 1)], [(3, 5), (0, 2), (5, 2), (1, 0), (2, 0), (1, 0), (0, 1), (3, 5)], [(3, 5), (3, 1), (3, 5), (3, 5), (4, 5), (1, 2), (2, 5), (5, 3)]]
batchAdd(game, sequences)

game='''
from typing import List, Any, Optional, Tuple
class System(AbstractSystem):
    def create_initial_board(self) -> List[int]:
        return [[1, 0, 0], [1, 1, 0], [0, 1, 1]]
        
    def get_legal_moves(self) -> List[Tuple[int, int]]:
        moves = []
        for i in range(1,5):
            for j in range(1,5):
                moves.append((i, j))
        return moves     
    
    def execute_move(self, move: Tuple[int, int]) -> None:
        i, j = move
        for x in range(3):
            for y in range(3):
                count=0
                for dx in range(-1, 2):
                    for dy in range(-1, 2):
                        count += self.board[(x + dx) % 3][(y + dy) % 3]
                if [i, j][self.board[x][y]] <= count:
                    self.board[x][y] = 1 - self.board[x][y]
'''
sequences=[[(1, 1), (4, 1)], [(1, 1), (4, 3)], [(1, 1), (4, 4)], [(2, 3), (4, 4)], [(1, 1), (3, 4)], [(1, 1), (4, 1), (1, 2), (1, 2)], [(1, 1), (4, 1), (1, 2), (1, 3)], [(1, 1), (4, 1), (1, 2), (1, 4)], [(1, 1), (4, 3), (1, 1), (1, 4)], [(1, 1), (4, 3), (1, 1), (4, 3)], [(1, 1), (4, 1), (1, 2), (1, 2), (1, 1), (1, 4)], [(1, 1), (4, 1), (1, 2), (1, 2), (1, 1), (2, 2)], [(1, 1), (4, 1), (1, 2), (1, 2), (1, 1), (3, 3)], [(1, 1), (4, 1), (1, 2), (1, 2), (1, 1), (4, 4)], [(1, 1), (4, 1), (1, 2), (1, 3), (1, 1), (1, 4)], [(1, 1), (4, 1), (1, 2), (1, 2), (1, 1), (1, 4), (1, 1), (1, 4)], [(1, 1), (4, 1), (1, 2), (1, 2), (1, 1), (1, 4), (1, 1), (3, 3)], [(1, 1), (4, 1), (1, 2), (1, 2), (1, 1), (1, 4), (1, 1), (4, 4)], [(1, 1), (4, 1), (1, 2), (1, 2), (1, 1), (3, 3), (1, 1), (1, 4)], [(1, 1), (4, 1), (1, 2), (1, 2), (1, 1), (3, 3), (1, 1), (3, 3)]]
batchAdd(game, sequences)


