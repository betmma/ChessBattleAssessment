#!/usr/bin/env python3
"""
Redundant Game Remover

This program removes redundant games by comparing their trajectories using a minimax agent.
Games are considered redundant if they produce identical trajectories (board states and rewards)
when played with the same agent configuration.

Usage:
    python redundant_game_remover.py --input_dir /path/to/games --output_dir /path/to/unique_games
"""

import sys
import os
import pkgutil
import json
import shutil
import hashlib
import argparse
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple, Set
from collections import defaultdict

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.universal_minimax_agent import UniversalMinimaxAgent
from utils.safe_json_dump import clean_np_types


class GameTrajectory:
    """Represents a game trajectory for comparison"""
    
    def __init__(self, board_states: List[str], rewards: List[float], final_winner: Any):
        self.board_states = board_states
        self.rewards = rewards
        self.final_winner = final_winner
        self._hash = None
    
    def __hash__(self):
        if self._hash is None:
            # Create a hash based on board states, rewards, and winner
            # Clean numpy types before JSON serialization
            content = json.dumps({
                'board_states': self.board_states,
                'rewards': [round(float(r), 3) for r in self.rewards],  # Round to avoid float precision issues
                'final_winner': clean_np_types(self.final_winner)
            }, sort_keys=True)
            self._hash = hashlib.md5(content.encode()).hexdigest()
        return hash(self._hash)
    
    def __eq__(self, other):
        if not isinstance(other, GameTrajectory):
            return False
        return (self.board_states == other.board_states and 
                all(abs(float(a) - float(b)) < 1e-6 for a, b in zip(self.rewards, other.rewards)) and
                clean_np_types(self.final_winner) == clean_np_types(other.final_winner))
    
    def to_dict(self):
        return {
            'board_states': self.board_states,
            'rewards': [float(r) for r in self.rewards],
            'final_winner': clean_np_types(self.final_winner)
        }


class TrajectoryCollector:
    """Collects game trajectories using minimax agent"""
    
    def __init__(self, depths: List[int] = [1, 2, 3], num_trajectories: int = 3):
        self.depths = depths
        self.num_trajectories = num_trajectories
        self.agents = {}
        
        # Create agents for each depth
        for depth in depths:
            self.agents[depth] = UniversalMinimaxAgent(
                name=f"Minimax_d{depth}",
                max_depth=depth,
                debug=False,
                same_return_random=False  # Deterministic behavior
            )
    
    def collect_trajectories(self, game_class) -> List[GameTrajectory]:
        """Collect trajectories for a game class using different agent depths"""
        trajectories = []
        
        for depth in self.depths:
            agent = self.agents[depth]
            
            # Collect multiple trajectories per depth (though they should be identical with same_return_random=False)
            for traj_idx in range(self.num_trajectories):
                try:
                    trajectory = self._play_game(game_class, agent)
                    if trajectory:
                        trajectories.append(trajectory)
                except Exception as e:
                    print(f"Error collecting trajectory for {game_class.name} with depth {depth}, trajectory {traj_idx}: {e}")
                    continue
        
        return trajectories
    
    def _play_game(self, game_class, agent) -> GameTrajectory:
        """Play a single game and collect the trajectory"""
        game = game_class()
        board_states = []
        rewards = []
        
        # Record initial state
        board_states.append(game.get_board_representation_for_llm())
        
        move_count = 0
        max_moves = 1000  # Prevent infinite games
        
        while not game.is_game_over() and move_count < max_moves:
            # Get current player's perspective
            current_player = game.get_current_player()
            
            # Get agent's move
            move_str = agent.get_move(game)
            
            # Parse the move back to the format expected by the game
            move = self._parse_move_string(move_str, game)
            if move is None:
                print(f"Failed to parse move: {move_str}")
                break
            
            # Calculate reward before making the move
            action_rewards = agent.get_action_rewards(game)
            move_reward = action_rewards.get(move_str, 0.0)
            rewards.append(move_reward)
            
            # Make the move
            success = game.make_move(move)
            if not success:
                print(f"Failed to make move: {move}")
                break
            
            # Record new board state
            board_states.append(game.get_board_representation_for_llm())
            
            move_count += 1
        
        # Get final winner
        final_winner = clean_np_types(game.check_winner())
        
        return GameTrajectory(board_states, rewards, final_winner)
    
    def _parse_move_string(self, move_str: str, game) -> Any:
        """Parse move string back to the format expected by the game"""
        try:
            # Try to parse as tuple first
            move = eval(move_str)
            return move
        except:
            return None


class RedundantGameRemover:
    """Main class for removing redundant games"""
    
    def __init__(self, input_dir: str, output_dir: str, depths: List[int] = [1, 2, 3]):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.depths = depths
        self.trajectory_collector = TrajectoryCollector(depths)
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Statistics
        self.total_games = 0
        self.unique_games = 0
        self.redundant_games = 0
    
    def load_game_classes(self) -> List[Any]:
        """Load all game classes from the input directory"""
        game_classes = []
        
        # Add input directory to Python path
        sys.path.insert(0, str(self.input_dir))
        
        try:
            for _, module_name, _ in pkgutil.iter_modules([str(self.input_dir)]):
                try:
                    game_class_name = module_name
                    module = __import__(game_class_name, fromlist=[game_class_name])
                    game_class_name=re.sub(r'gen_\d+_','',game_class_name)
                    game_class = getattr(module, game_class_name)
                    game_classes.append(game_class)
                    print(f"Loaded game class: {game_class_name}")
                except Exception as e:
                    print(f"Failed to load game class {module_name}: {e}")
                    continue
        finally:
            # Remove from path to avoid conflicts
            if str(self.input_dir) in sys.path:
                sys.path.remove(str(self.input_dir))
        
        return game_classes
    
    def remove_redundant_games(self) -> Dict[str, Any]:
        """Main method to remove redundant games"""
        print(f"Loading games from {self.input_dir}")
        game_classes = self.load_game_classes()
        self.total_games = len(game_classes)
        
        print(f"Found {self.total_games} games")
        
        if self.total_games == 0:
            print("No games found!")
            return self._get_statistics()
        
        # Group games by their trajectories
        trajectory_groups = defaultdict(list)
        game_trajectories = {}
        
        print("Collecting trajectories...")
        for i, game_class in enumerate(game_classes):
            print(f"Processing game {i+1}/{self.total_games}: {game_class.name}")
            
            try:
                trajectories = self.trajectory_collector.collect_trajectories(game_class)
                
                if not trajectories:
                    print(f"  No trajectories collected for {game_class.name}")
                    continue
                
                # Use the first trajectory as representative
                representative_trajectory = trajectories[0]
                game_trajectories[game_class] = trajectories
                
                # Group by trajectory hash
                trajectory_hash = hash(representative_trajectory)
                trajectory_groups[trajectory_hash].append(game_class)
                
                print(f"  Collected {len(trajectories)} trajectories")
                
            except Exception as e:
                print(f"  Error processing {game_class.name}: {e}")
                continue
        
        # Find unique games (keep the first game from each group)
        unique_games = []
        redundant_games = []
        
        print("\nIdentifying unique games...")
        for trajectory_hash, games_in_group in trajectory_groups.items():
            if len(games_in_group) == 1:
                unique_games.extend(games_in_group)
                print(f"  Unique: {games_in_group[0].name}")
            else:
                # Keep the first game, mark others as redundant
                unique_games.append(games_in_group[0])
                redundant_games.extend(games_in_group[1:])
                print(f"  Group of {len(games_in_group)} redundant games:")
                print(f"    Keeping: {games_in_group[0].name}")
                for redundant_game in games_in_group[1:]:
                    print(f"    Removing: {redundant_game.name}")
        
        self.unique_games = len(unique_games)
        self.redundant_games = len(redundant_games)
        
        # Copy unique games to output directory
        print(f"\nCopying {self.unique_games} unique games to {self.output_dir}")
        self._copy_unique_games(unique_games)
        
        # Save detailed analysis
        self._save_analysis(trajectory_groups, game_trajectories)
        
        return self._get_statistics()
    
    def _copy_unique_games(self, unique_games: List[Any]):
        """Copy unique game files to output directory"""
        from inspect import getmodule
        for game_class in unique_games:
            source_file = getmodule(game_class).__file__
            dest_file = self.output_dir / f"{game_class.__name__}.py"
            
            try:
                shutil.copy2(source_file, dest_file)
                print(f"  Copied: {game_class.__name__}.py")
            except Exception as e:
                print(f"  Failed to copy {game_class.__name__}.py: {e}")
    
    def _save_analysis(self, trajectory_groups: Dict, game_trajectories: Dict):
        """Save detailed analysis to JSON file"""
        analysis = {
            'summary': self._get_statistics(),
            'redundant_groups': [],
            'trajectory_details': {}
        }
        
        # Record redundant groups
        for trajectory_hash, games_in_group in trajectory_groups.items():
            if len(games_in_group) > 1:
                group_info = {
                    'trajectory_hash': str(trajectory_hash),
                    'kept_game': games_in_group[0].__name__,
                    'removed_games': [g.__name__ for g in games_in_group[1:]],
                    'total_games_in_group': len(games_in_group)
                }
                analysis['redundant_groups'].append(group_info)
        
        # Record trajectory details for unique games (sample)
        for game_class, trajectories in list(game_trajectories.items())[:10]:  # Limit to first 10 for size
            if trajectories:
                analysis['trajectory_details'][game_class.__name__] = {
                    'num_trajectories': len(trajectories),
                    'sample_trajectory': trajectories[0].to_dict()
                }
        
        # Save analysis
        analysis_file = self.output_dir / 'redundancy_analysis.json'
        with open(analysis_file, 'w') as f:
            json.dump(analysis, f, indent=2, default=lambda x: clean_np_types(x))
        
        print(f"Saved analysis to {analysis_file}")
    
    def _get_statistics(self) -> Dict[str, Any]:
        """Get removal statistics"""
        return {
            'total_games': self.total_games,
            'unique_games': self.unique_games,
            'redundant_games': self.redundant_games,
            'removal_rate': f"{(self.redundant_games / max(self.total_games, 1)) * 100:.1f}%",
            'depths_used': self.depths
        }


def main():
    parser = argparse.ArgumentParser(description='Remove redundant games based on trajectory comparison')
    parser.add_argument('--input_dir', type=str, default='/root/myr/genGames/0803/successGames',
                       help='Directory containing generated games')
    parser.add_argument('--output_dir', type=str, default='/root/myr/genGames/0803/uniqueGames',
                       help='Directory to save unique games')
    parser.add_argument('--depths', type=int, nargs='+', default=[1],
                       help='Minimax depths to use for trajectory collection')
    parser.add_argument('--num_trajectories', type=int, default=3,
                       help='Number of trajectories to collect per depth')
    
    args = parser.parse_args()
    
    print("=== Redundant Game Remover ===")
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Minimax depths: {args.depths}")
    print(f"Trajectories per depth: {args.num_trajectories}")
    print()
    
    # Create remover and process games
    remover = RedundantGameRemover(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        depths=args.depths
    )
    
    # Update trajectory collector with command line arguments
    remover.trajectory_collector.num_trajectories = args.num_trajectories
    
    try:
        statistics = remover.remove_redundant_games()
        
        print("\n=== Summary ===")
        print(f"Total games processed: {statistics['total_games']}")
        print(f"Unique games kept: {statistics['unique_games']}")
        print(f"Redundant games removed: {statistics['redundant_games']}")
        print(f"Removal rate: {statistics['removal_rate']}")
        print(f"Unique games saved to: {args.output_dir}")
        
    except Exception as e:
        print(f"Error during processing: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
