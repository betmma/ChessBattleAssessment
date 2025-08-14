import json
import os
import sys
import argparse
from tqdm import tqdm
import multiprocessing as mp

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from games import GameByName, Games
from agents.universal_minimax_agent import UniversalMinimaxAgent

# --- Multiprocessing helpers ---
_MINIMAX_AGENT = None
_MAX_DEPTH = None

def _init_minimax_worker(max_depth: int):
    """Initializer for each worker process to create a Minimax agent once."""
    global _MINIMAX_AGENT, _MAX_DEPTH
    _MAX_DEPTH = max_depth
    if max_depth and max_depth > 0:
        _MINIMAX_AGENT = UniversalMinimaxAgent(max_depth=max_depth)
    else:
        _MINIMAX_AGENT = None


def _compute_entry_for_board(task):
    """Worker function to compute action rewards and prompts for a single board.
    Args:
        task: Tuple (game_name: str, board_str: str)
    Returns:
        dataset_entry dict or None if failed/filtered.
    """
    game_name, board_str = task
    try:
        # Construct game and load state
        game_class = GameByName(game_name)
        game = game_class()
        game.load_state_from_representation(board_str)

        # If depth disabled, skip (caller should handle non-MP path in this case)
        if not _MAX_DEPTH or _MAX_DEPTH <= 0:
            return None

        # Compute rewards using per-process agent (fallback to on-demand if missing)
        agent = _MINIMAX_AGENT or UniversalMinimaxAgent(max_depth=_MAX_DEPTH)
        action_rewards = agent.get_action_rewards(game)

        # Filter rewards to ensure quality
        from ChessBattleAssessment.generate_grpo_dataset import filter_rewards  # lazy import to avoid circular on spawn
        if not filter_rewards(action_rewards):
            return None

        # Build prompts with the same agent context
        prompts = game.get_chat_history_for_llm(agent)

        return {
            "prompt": prompts,
            "task": game_name,
            "reward_model": {"ground_truth": action_rewards},
        }
    except Exception:
        # Silently ignore failures in worker; main process logs counts
        return None


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
        
        # If max_depth <= 0, keep original sequential path
        if max_depth <= 0:
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
                            board_key = f"{game_name}:{board_str}"
                            if board_key in processed_boards:
                                continue
                            processed_boards.add(board_key)
                            game = game_class()
                            try:
                                game.load_state_from_representation(board_str)
                            except Exception as e:
                                print(f"Warning: Failed to load game state for {game_name} game {game_id}: {e}")
                                continue
                            action_rewards = move_record.get('action_rewards', [])
                            if not action_rewards:
                                continue
                            if not filter_rewards(action_rewards):
                                continue
                            try:
                                prompts = game.get_chat_history_for_llm(minimax_agent)
                            except Exception as e:
                                print(f"Warning: Failed to get chat history for {game_name}: {e}")
                                continue
                            dataset_entry = {
                                "prompt": prompts,
                                "task": game_name,
                                "reward_model": {"ground_truth": action_rewards},
                            }
                            f_out.write(json.dumps(dataset_entry) + '\n')
                            processed_count += 1
                            pbar.set_description(f"Processing moves (Generated: {processed_count} entries)")
            print(f"\nTotal unique board states processed: {processed_count}")
            return

        # For max_depth > 0: build unique board tasks then process in parallel
        tasks = []
        for game_name, game_data in detailed_logs.items():
            games = game_data.get("games", {})
            for game_id, game_record in games.items():
                moves = game_record.get("moves", [])
                for move_record in moves:
                    board_str = move_record.get("board_before", "")
                    if not board_str:
                        continue
                    board_key = f"{game_name}:{board_str}"
                    if board_key in processed_boards:
                        continue
                    processed_boards.add(board_key)
                    tasks.append((game_name, board_str))

        processed_count = 0
        skipped_count = 0
        num_workers = max(1, (os.cpu_count() or 1))

        with tqdm(total=len(tasks), desc="Processing boards", unit="board") as pbar:
            with mp.Pool(processes=num_workers, initializer=_init_minimax_worker, initargs=(max_depth,)) as pool:
                # Use a reasonable chunksize for better throughput
                for result in pool.imap_unordered(_compute_entry_for_board, tasks, chunksize=16):
                    pbar.update(1)
                    if result is None:
                        skipped_count += 1
                        continue
                    f_out.write(json.dumps(result) + '\n')
                    processed_count += 1
                    pbar.set_description(f"Processing boards (Generated: {processed_count}, Skipped: {skipped_count})")
        
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
    input_file = '/remote-home1/yrmou/ChessBattleAssessment/evaluation_results_vllm/game_logs/20250813-044003_VLLMAgent_vs_VLLMAgent_CONSOLIDATED.json'
    output_file = 'evaluation_results_vllm/grpo/DrCoNi_lv2_raw2_d4.jsonl'
    max_depth = 4

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
