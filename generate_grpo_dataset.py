import json
import os
import sys
import argparse
from tqdm import tqdm

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
        
        # Count total moves for progress bar
        total_moves = 0
        for game_name, game_data in detailed_logs.items():
            games = game_data.get("games", {})
            for game_id, game_record in games.items():
                total_moves += len(game_record.get("moves", []))
        
        processed_count = 0
        with tqdm(total=total_moves, desc="Processing moves", unit="move") as pbar:
            for game_name, game_data in detailed_logs.items():
                print(f"\nProcessing {game_name} games...")
                
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
                        pbar.update(1)
                        board_str = move_record.get("board_before", "")
                        
                        if not board_str:
                            continue
                        
                        # Create a unique, hashable representation of the board state
                        board_key = f"{game_name}:{board_str}"
                        if board_key in processed_boards:
                            continue  # Skip if this board state has been processed
                        
                        processed_boards.add(board_key)
                        processed_count += 1
                        
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
                            action_rewards = minimax_agent.get_action_rewards(game) if max_depth>0 else move_record.get('action_rewards', [])
                        except Exception as e:
                            print(f"Warning: Failed to get action rewards for {game_name}: {e}")
                            continue
                        
                        if not filter_rewards(action_rewards): continue
                        
                        # Get the prompts
                        try:
                            prompts = game.get_chat_history_for_llm(minimax_agent)
                        except Exception as e:
                            print(f"Warning: Failed to get chat history for {game_name}: {e}")
                            continue
                        
                        # Construct the dataset entry
                        dataset_entry = {
                            "prompt": prompts,
                            "task": game_name,
                            "reward_model": {
                                "ground_truth": action_rewards
                            }
                        }
                        
                        f_out.write(json.dumps(dataset_entry) + '\n')
                        
                        # Update progress bar description with current stats
                        pbar.set_description(f"Processing moves (Generated: {processed_count} entries)")
        
        print(f"\nTotal unique board states processed: {processed_count}")

def recalculate_rewards_from_jsonl(input_jsonl: str, output_jsonl: str, new_max_depth: int = 6):
    """
    Reads a generated JSONL dataset and recalculates rewards using a different minimax depth.
    
    Args:
        input_jsonl: Path to the input JSONL file with existing dataset entries.
        output_jsonl: Path to the output JSONL file with updated rewards.
        new_max_depth: The new maximum depth for the Minimax agent.
    """
    minimax_agent = UniversalMinimaxAgent(max_depth=new_max_depth)
    processed_count = 0
    skipped_count = 0
    
    # Count total entries for progress bar
    total_entries = 0
    with open(input_jsonl, 'r') as f:
        for _ in f:
            total_entries += 1
    
    with open(input_jsonl, 'r') as f_in, open(output_jsonl, 'w') as f_out:
        with tqdm(total=total_entries, desc="Recalculating rewards", unit="entry") as pbar:
            for line in f_in:
                pbar.update(1)
                
                try:
                    entry = json.loads(line.strip())
                except json.JSONDecodeError as e:
                    print(f"Warning: Failed to parse JSON line: {e}")
                    skipped_count += 1
                    continue
                
                task = entry.get("task", "")
                prompt = entry.get("prompt", [])
                
                if not task or not prompt:
                    print(f"Warning: Missing task or prompt in entry, skipping...")
                    skipped_count += 1
                    continue
                
                # Extract board state from the prompt
                # The board state should be in the user message content
                board_str = None
                for message in prompt:
                    if message.get("role") == "user":
                        content = message.get("content", "")
                        # Extract the board portion (everything before "You are player")
                        if "You are player" in content:
                            board_str = content.split("You are player")[0].strip()
                            break
                
                if not board_str:
                    print(f"Warning: Could not extract board state from prompt, skipping...")
                    skipped_count += 1
                    continue
                
                # Get the game class and create instance
                try:
                    game_class = GameByName(task)
                    game = game_class()
                except KeyError:
                    print(f"Warning: Unknown game type {task}, skipping...")
                    skipped_count += 1
                    continue
                
                # Load the game state
                try:
                    game.load_state_from_representation(board_str)
                except Exception as e:
                    print(f"Warning: Failed to load game state for {task}: {e}")
                    skipped_count += 1
                    continue
                
                # Recalculate action rewards with new depth
                try:
                    new_action_rewards = minimax_agent.get_action_rewards(game)
                except Exception as e:
                    print(f"Warning: Failed to get action rewards for {task}: {e}")
                    skipped_count += 1
                    continue
                
                # Filter rewards to ensure quality
                if not filter_rewards(new_action_rewards):
                    skipped_count += 1
                    continue
                
                # Update the entry with new rewards
                updated_entry = entry.copy()
                updated_entry["reward_model"]["ground_truth"] = new_action_rewards
                
                # Write the updated entry
                f_out.write(json.dumps(updated_entry) + '\n')
                processed_count += 1
                
                # Update progress bar description
                pbar.set_description(f"Recalculating rewards (Processed: {processed_count}, Skipped: {skipped_count})")
    
    print(f"\nRecalculation complete!")
    print(f"Total entries processed: {processed_count}")
    print(f"Total entries skipped: {skipped_count}")
    print(f"Updated dataset saved to: {output_jsonl}")

def filter_rewards(action_rewards):
    """
    Filters out good rewards 
    """
    reward_values=list(action_rewards.values())
    
    if len(reward_values) <=1: 
        return False
    
    if all(i==reward_values[0] for i in reward_values):
        return False
    
    win_in_n_threshold=990
    if (any(i>win_in_n_threshold for i in reward_values) or any(i<-win_in_n_threshold for i in reward_values)):
        # If there is a win/lose in n moves
        return True

    return False

if __name__ == '__main__':
    input_file = 'evaluation_results_vllm/grpo/DrCoNi_1000.jsonl'
    output_file = 'evaluation_results_vllm/grpo/DrCoNi_1000_d6.jsonl'
    max_depth = 6

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
    
    if input_file.endswith('.json'):
        generate_dataset(input_file, output_file, max_depth)
    elif input_file.endswith('.jsonl'):
        recalculate_rewards_from_jsonl(input_file, output_file, new_max_depth=max_depth)
    print(f"Dataset generated successfully at {output_file}")
