import json
import os
import sys
import argparse

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from games import GameByName, Games
from agents.universal_minimax_agent import UniversalMinimaxAgent

def generate_dataset(input_file: str, output_file: str, max_depth: int = 4):
    """
    Generates a GRPO dataset from consolidated game logs.

    Args:
        input_file: Path to the input consolidated JSON file with game logs.
        output_file: Path to the output JSONL file for the dataset.
        max_depth: The maximum depth for the Minimax agent.
    """
    minimax_agent = UniversalMinimaxAgent(max_depth=max_depth)
    processed_boards = set()
    
    with open(input_file, 'r') as f_in:
        consolidated_data = json.load(f_in)
    
    with open(output_file, 'w') as f_out:
        # Process each game type in the consolidated logs
        detailed_logs = consolidated_data.get("detailed_logs", {})
        
        for game_name, game_data in detailed_logs.items():
            print(f"Processing {game_name} games...")
            
            # Get the game class
            try:
                game_class = GameByName(game_name)
            except KeyError:
                print(f"Warning: Unknown game type {game_name}, skipping...")
                continue
            
            games = game_data.get("games", {})
            
            for game_id, game_record in games.items():
                moves = game_record.get("moves", [])
                
                for move_record in moves:
                    board_str = move_record.get("board_before", "")
                    
                    if not board_str:
                        continue
                    
                    # Create a unique, hashable representation of the board state
                    board_key = f"{game_name}:{board_str}"
                    if board_key in processed_boards:
                        continue  # Skip if this board state has been processed
                    
                    processed_boards.add(board_key)
                    
                    # Create game instance and load state from representation
                    game = game_class()
                    
                    try:
                        # Use the game's built-in method to load state from string representation
                        game.load_state_from_representation(board_str)
                    except Exception as e:
                        print(f"Warning: Failed to load game state for {game_name} game {game_id}: {e}")
                        continue
                    
                    # Get rewards for all possible moves using the universal minimax agent
                    try:
                        action_rewards = minimax_agent.get_action_rewards(game)
                    except Exception as e:
                        print(f"Warning: Failed to get action rewards for {game_name}: {e}")
                        continue
                    
                    # Get the prompts
                    try:
                        prompts = game.get_chat_history_for_llm(minimax_agent)
                    except Exception as e:
                        print(f"Warning: Failed to get chat history for {game_name}: {e}")
                        continue
                    
                    # Construct the dataset entry
                    dataset_entry = {
                        "prompt": prompts,
                        "reward_model": {
                            "ground_truth": action_rewards
                        }
                    }
                    
                    f_out.write(json.dumps(dataset_entry) + '\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate GRPO dataset from consolidated game logs")
    parser.add_argument("--input", type=str, required=True, help="Path to consolidated JSON log file")
    parser.add_argument("--output", type=str, required=True, help="Path to output JSONL dataset file")
    parser.add_argument("--max_depth", type=int, default=4, help="Maximum depth for minimax agent")
    
    args = parser.parse_args()
    
    # Check if running as script with default values (for backwards compatibility)
    if len(sys.argv) == 1:
        # Use default values for backwards compatibility
        input_file = 'evaluation_results_vllm/game_logs/CONSOLIDATED_Minimax-random-0.0-depth-4_vs_RandomAgent_20250706-182632.json'
        output_file = 'evaluation_results_vllm/grpo/grpo_dataset.jsonl'
        max_depth = 4
    else:
        input_file = args.input
        output_file = args.output
        max_depth = args.max_depth
    
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} does not exist")
        print("Available consolidated log files:")
        log_dir = 'evaluation_results_vllm/game_logs'
        if os.path.exists(log_dir):
            for file in os.listdir(log_dir):
                if file.startswith('CONSOLIDATED_') and file.endswith('.json'):
                    print(f"  {os.path.join(log_dir, file)}")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    generate_dataset(input_file, output_file, max_depth)
    print(f"Dataset generated successfully at {output_file}")
