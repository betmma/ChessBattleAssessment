import copy
import random
from collections import deque
from typing import List, Any, Tuple

# ================= CONFIGURATION =================
# How many sequences of each length do you want?
TARGET_COUNTS = {2: 5, 4: 5, 6: 5, 8: 5}
TARGET_COUNTS = {i:5 for i in range(1,6)}

# ================= HELPER FUNCTIONS =================

def to_hashable(obj):
    """Recursively converts lists to tuples for hashing in visited set."""
    if isinstance(obj, list):
        return tuple(to_hashable(x) for x in obj)
    return obj

def generate_sequences_for_game(game_class, game_name):
    print(f"--- Generating sequences for {game_name} ---")
    
    # Storage for results: {length: [list_of_moves]}
    found = {k: [] for k in TARGET_COUNTS.keys()}
    max_depth = max(TARGET_COUNTS.keys())
    
    # BFS Initialization
    initial_game = game_class()
    initial_state = to_hashable(initial_game.board)
    
    # Queue: (game_instance, path_taken)
    queue = deque([(initial_game, [])])
    visited = {initial_state}
    
    while queue:
        current_game, path = queue.popleft()
        depth = len(path)
        
        # If this path length is one we need, store it
        if depth in TARGET_COUNTS and len(found[depth]) < TARGET_COUNTS[depth]:
            found[depth].append(path)
            # Check if we are completely done
            if all(len(found[k]) >= TARGET_COUNTS[k] for k in TARGET_COUNTS):
                break
        
        # Stop expanding if we've reached the max depth needed
        if depth >= max_depth:
            continue
            
        # Get legal moves and shuffle them to ensure random selection 
        # (otherwise BFS always picks the 'first' legal move)
        moves = current_game.get_legal_moves()
        random.shuffle(moves)
        
        for move in moves:
            # Create a completely new copy to execute the move
            next_game = copy.deepcopy(current_game)
            try:
                next_game.execute_move(move)
                next_state = to_hashable(next_game.board)
                
                if next_state not in visited:
                    visited.add(next_state)
                    queue.append((next_game, path + [move]))
            except Exception:
                # In case a move becomes invalid due to game logic quirks
                continue

    # Format Output
    all_sequences = []
    for length in sorted(found.keys()):
        for seq in found[length]:
            all_sequences.append(seq)
            
    print(f'len(sequences) = {len(all_sequences)}')
    print(f"sequences = {all_sequences}")
    print("\n")


# ================= GAME DEFINITIONS =================
# (Pasted exactly from your samples)

class AbstractSystem:
    def __init__(self):
        self.board = self.create_initial_board()
    def create_initial_board(self): pass
    def get_legal_moves(self): pass
    def execute_move(self, move): pass

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
            
# ================= EXECUTION =================

if __name__ == "__main__":
    generate_sequences_for_game(System, "??")