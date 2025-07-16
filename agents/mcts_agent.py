import sys
import os
import random
import math
import time
from typing import List, Dict, Any, Optional
from collections import defaultdict

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.agent import Agent

class MCTSNode:
    """A node in the MCTS tree"""
    
    def __init__(self, game_state, move=None, parent=None):
        self.game_state = game_state.clone() if game_state else None
        self.move = move  # The move that led to this state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.wins = 0.0
        self.untried_moves = None  # Will be populated with legal moves
        self.is_terminal = False
        
        if game_state:
            self.is_terminal = game_state.is_game_over()
            if not self.is_terminal:
                self.untried_moves = list(game_state.get_legal_moves())
            else:
                self.untried_moves = []
    
    def is_fully_expanded(self):
        """Check if all possible moves from this node have been tried"""
        return len(self.untried_moves) == 0
    
    def add_child(self, move, game_state):
        """Add a new child node for the given move"""
        child = MCTSNode(game_state, move, self)
        self.children.append(child)
        if move in self.untried_moves:
            self.untried_moves.remove(move)
        return child
    
    def get_ucb1_value(self, exploration_param=math.sqrt(2)):
        """Calculate UCB1 value for node selection"""
        if self.visits == 0:
            return float('inf')
        
        exploitation = self.wins / self.visits
        exploration = exploration_param * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploitation + exploration
    
    def select_best_child(self, exploration_param=math.sqrt(2)):
        """Select the child with the highest UCB1 value"""
        return max(self.children, key=lambda child: child.get_ucb1_value(exploration_param))
    
    def update(self, result):
        """Update this node with simulation result"""
        self.visits += 1
        self.wins += result
    
    def is_root(self):
        """Check if this is the root node"""
        return self.parent is None

class MCTSAgent(Agent):
    """Monte Carlo Tree Search agent for board games"""
    
    def __init__(self, name: str = "MCTS", 
                 simulations: int = 1000, 
                 exploration_param: float = math.sqrt(2),
                 time_limit: float = None,
                 random_rollouts: bool = True,
                 use_prior_knowledge: bool = False):
        """
        Initialize MCTS agent
        
        Args:
            name: Agent name
            simulations: Number of MCTS simulations to run
            exploration_param: UCB1 exploration parameter (sqrt(2) is theoretically optimal)
            time_limit: Time limit in seconds (overrides simulations if set)
            random_rollouts: Whether to use random rollouts or basic heuristics
            use_prior_knowledge: Whether to use game-specific knowledge for node evaluation
        """
        super().__init__(name)
        self.simulations = simulations
        self.exploration_param = exploration_param
        self.time_limit = time_limit
        self.random_rollouts = random_rollouts
        self.use_prior_knowledge = use_prior_knowledge
        
    def get_move(self, game) -> str:
        """
        Get best move using MCTS
        
        Args:
            game: Game object
            
        Returns:
            str: The chosen move as a string
        """
        if game.is_game_over():
            return "Game is over"
        
        legal_moves = game.get_legal_moves()
        if not legal_moves:
            return "No legal moves"
        
        if len(legal_moves) == 1:
            return str(legal_moves[0])
        
        # Set the current player for evaluation
        self.current_player = game.get_current_player()
        
        # Run MCTS
        root = MCTSNode(game)
        
        start_time = time.time()
        simulations_run = 0
        
        while True:
            # Check termination conditions
            if self.time_limit:
                if time.time() - start_time >= self.time_limit:
                    break
            else:
                if simulations_run >= self.simulations:
                    break
            
            # Run one MCTS iteration
            self._mcts_iteration(root)
            simulations_run += 1
        
        # Select best move based on visit count (most robust)
        if not root.children:
            return str(legal_moves[0])  # Fallback
            
        best_child = max(root.children, key=lambda child: child.visits)
        return str(best_child.move)
    
    def _mcts_iteration(self, root):
        """Run one iteration of MCTS: Selection -> Expansion -> Simulation -> Backpropagation"""
        
        # 1. Selection: traverse down the tree using UCB1
        node = self._select(root)
        
        # 2. Expansion: add a new child node if possible
        if not node.is_terminal and not node.is_fully_expanded():
            node = self._expand(node)
        
        # 3. Simulation: play out the game randomly from this node
        result = self._simulate(node)
        
        # 4. Backpropagation: update all nodes on the path
        self._backpropagate(node, result)
    
    def _select(self, node):
        """Selection phase: traverse tree using UCB1 until we reach a leaf"""
        while not node.is_terminal and node.is_fully_expanded():
            node = node.select_best_child(self.exploration_param)
        return node
    
    def _expand(self, node):
        """Expansion phase: add a new child node"""
        if node.untried_moves:
            # Select a random untried move
            move = random.choice(node.untried_moves)
            
            # Create new game state
            new_game = node.game_state.clone()
            new_game.make_move(move)
            
            # Add child node
            child = node.add_child(move, new_game)
            return child
        return node
    
    def _simulate(self, node):
        """Simulation phase: play out the game randomly from the given node"""
        game = node.game_state.clone()
        
        # Random rollout
        while not game.is_game_over():
            legal_moves = game.get_legal_moves()
            if not legal_moves:
                break
            
            if self.random_rollouts:
                move = random.choice(legal_moves)
            else:
                # Use simple heuristics if available
                move = self._select_heuristic_move(game, legal_moves)
            
            game.make_move(move)
        
        # Return result from perspective of the current player at the root
        return self._evaluate_terminal_state(game)
    
    def _select_heuristic_move(self, game, legal_moves):
        """Select move using simple heuristics instead of pure random"""
        # If game has position evaluation, use it
        if hasattr(game, 'evaluate_position'):
            best_move = None
            best_score = float('-inf')
            current_player = game.get_current_player()
            
            for move in legal_moves:
                temp_game = game.clone()
                temp_game.make_move(move)
                
                # Quick terminal check
                if temp_game.is_game_over():
                    winner = temp_game.check_winner()
                    if winner == current_player:
                        return move  # Winning move
                
                score = temp_game.evaluate_position()
                if current_player == -1:
                    score = -score
                
                if score > best_score:
                    best_score = score
                    best_move = move
            
            return best_move if best_move is not None else random.choice(legal_moves)
        
        return random.choice(legal_moves)
    
    def _evaluate_terminal_state(self, game):
        """Evaluate the terminal state, returning 1 for current player win, 0 for loss, 0.5 for draw"""
        if not game.is_game_over():
            return 0.5  # Should not happen, but safe default
        
        winner = game.check_winner()
        
        if winner is None or winner == 0:
            return 0.5  # Draw
        elif winner == self.current_player:
            return 1.0  # Win for the player we're optimizing for
        else:
            return 0.0  # Loss for the player we're optimizing for
    
    def _backpropagate(self, node, result):
        """Backpropagation phase: update all nodes from leaf to root"""
        current_result = result
        current_node = node
        
        while current_node is not None:
            current_node.update(current_result)
            # Flip result for parent since it represents the opponent's perspective
            current_result = 1.0 - current_result
            current_node = current_node.parent
    
    def get_action_rewards(self, game) -> Dict[str, float]:
        """
        Get action rewards for all legal moves (for analysis/training)
        
        Args:
            game: Game object
            
        Returns:
            Dict[str, float]: Dictionary mapping moves to their estimated values
        """
        legal_moves = game.get_legal_moves()
        if not legal_moves:
            return {}
        
        # Set the current player for evaluation
        self.current_player = game.get_current_player()
        
        # Run MCTS to build the tree
        root = MCTSNode(game)
        
        start_time = time.time()
        simulations_run = 0
        
        while True:
            if self.time_limit:
                if time.time() - start_time >= self.time_limit:
                    break
            else:
                if simulations_run >= self.simulations:
                    break
            
            self._mcts_iteration(root)
            simulations_run += 1
        
        # Calculate rewards based on visit counts and win rates
        rewards = {}
        total_visits = sum(child.visits for child in root.children)
        
        for child in root.children:
            if child.visits > 0:
                win_rate = child.wins / child.visits
                # Combine win rate with visit proportion for confidence
                confidence = child.visits / total_visits if total_visits > 0 else 0
                reward = win_rate * 0.8 + confidence * 0.2
            else:
                reward = 0.5  # Unknown
            
            rewards[str(child.move)] = reward
        
        # Add unvisited moves with low reward
        for move in legal_moves:
            if str(move) not in rewards:
                rewards[str(move)] = 0.1
        
        return rewards
    
    def set_simulations(self, simulations: int):
        """Set number of simulations"""
        self.simulations = simulations
    
    def set_time_limit(self, time_limit: float):
        """Set time limit in seconds"""
        self.time_limit = time_limit
    
    def set_exploration_param(self, exploration_param: float):
        """Set UCB1 exploration parameter"""
        self.exploration_param = exploration_param
