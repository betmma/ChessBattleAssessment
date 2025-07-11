import json
import re

def reformat_board_and_moves_in_content(content_str: str) -> str:
    """
    Finds a Connect 4 board and legal moves list in the content string,
    reformats them, and returns the modified content string.

    The board is changed to a (row, col) coordinate format, e.g., (0,0): "R".
    The legal moves are annotated with the landing position of the piece.
    """
    lines = content_str.splitlines()
    
    # Constants for parsing
    board_data_rows = 6
    board_columns_header_prefix = "Columns:"
    legal_moves_prefix = "Your available legal moves (columns):"

    # --- Step 1: Pre-parse the board into a grid ---
    # This is necessary because the legal moves section needs the grid data.
    grid = []
    board_header_index = -1
    for i, line in enumerate(lines):
        if line.strip().startswith(board_columns_header_prefix):
            board_header_index = i
            break
    
    if board_header_index != -1:
        # We found a board, let's parse it
        board_data_start_index = board_header_index + 1
        board_data_end_index = board_data_start_index + board_data_rows
        board_lines_str = lines[board_data_start_index:board_data_end_index]
        
        for r_str in board_lines_str:
            # Split by space and filter out empty strings from multiple spaces
            cells = [cell for cell in r_str.strip().split(' ') if cell]
            grid.append(cells)

        # Basic validation to ensure board is not malformed
        if not grid or not all(len(row) == len(grid[0]) for row in grid):
            print("Warning: Malformed or non-rectangular board found. Aborting reformat.")
            return content_str
    else:
        # No board found, no reformatting to do
        return content_str

    # --- Step 2: Iterate through lines and build the new content ---
    new_content_lines = []
    i = 1
    while i < len(lines):
        line = lines[i]
        stripped_line = line.strip()

        if i == board_header_index:
            # We are at the old board header. Replace it and the old board
            # with the new coordinate-based format.
            new_content_lines.append("Current Connect 4 board state (row, col):")
            
            num_rows = len(grid)
            num_cols = len(grid[0])
            for r in range(num_rows):
                row_parts = [f'({r},{c}): "{grid[r][c]}"' for c in range(num_cols)]
                new_content_lines.append(", ".join(row_parts))
            
            # Skip past the original header and board rows in the input
            i += (1 + board_data_rows)
            
        elif stripped_line.startswith(legal_moves_prefix):
            # We are at the legal moves line.
            # Keep the original line, then add the landing spot info.
            new_content_lines.append(line)

            # Extract legal move numbers from a string like "...: [0, 1, 5]"
            match = re.search(r'\[(.*?)\]', stripped_line)
            if match:
                moves_str = match.group(1)
                if moves_str: # Ensure the list is not empty
                    legal_moves = [int(m.strip()) for m in moves_str.split(',') if m.strip()]
                    
                    num_rows = len(grid)
                    for col in legal_moves:
                        landing_row = -1
                        # Find the first empty spot ('.') from the bottom up
                        for r in range(num_rows - 1, -1, -1):
                            # Check column bounds and if spot is empty
                            if col < len(grid[r]) and grid[r][col] == '.':
                                landing_row = r
                                break
                        
                        if landing_row != -1:
                            new_content_lines.append(f"Column {col} will land at ({landing_row}, {col})")
                        else:
                            # This case should not happen for a valid legal move
                            print(f"Warning: Column {col} is full but listed as a legal move.")

            i += 1 # Move to the next line in the input

        else:
            # This is any other line, just copy it over.
            new_content_lines.append(line)
            i += 1
            
    return "\n".join(new_content_lines)


def process_jsonl_file(input_filepath: str, output_filepath: str):
    """
    Reads a JSONL file, processes the Connect 4 board in specified fields,
    and writes the modified data to a new JSONL file.
    """
    rewards=[]
    
    with open(input_filepath, 'r', encoding='utf-8') as infile, \
         open(output_filepath, 'w', encoding='utf-8') as outfile:
        
        for line_num, line_str in enumerate(infile):
            try:
                data = json.loads(line_str)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON on line {line_num + 1}: {e}")
                outfile.write(line_str) # Write original line if it's broken
                continue

            # Only process entries that are a "win in 1" scenario
            # if 999 not in data.get('reward_model', {}).get("ground_truth", []):
            #     continue
                
            if isinstance(data.get("prompt"), list) and len(data["prompt"]) > 1:
                if isinstance(data["prompt"][1], dict) and "content" in data["prompt"][1]:
                    original_content = data["prompt"][1]["content"]
                    # Use the new reformatting function
                    modified_content = reformat_board_and_moves_in_content(original_content)
                    data["prompt"][1]["content"] = modified_content
                    
            data["reward_model"]["ground_truth"] = (list(data["reward_model"]["ground_truth"].values())) #reward_rescale
            rewards.append(data["reward_model"]["ground_truth"])
            
            outfile.write(json.dumps(data) + '\n')

    print(f"Processing complete. Output written to {output_filepath}")
    reward_statistics(rewards)
def reward_statistics(rewards: list) -> None:
    ave=lambda x: sum(x)/len(x) if x else 0
    print(f"Average reward: {ave([ave(r) for r in rewards if r is not None])}")
    print(f"Max reward average: {ave([max(r) for r in rewards if r is not None])}")
    print(f"Min reward average: {ave([min(r) for r in rewards if r is not None])}")

def reward_rescale(reward: list) -> list:
    middle=[i for i in reward if -120<=i<=120]
    mini,maxi=-20,20
    if middle:
        mini, maxi = min(middle), max(middle)
    rescaled = []
    for r in reward:
        if r<-1000:
            rescaled.append(mini-20)
        elif r>1000:
            rescaled.append(maxi+20)
        elif r < mini:
            rescaled.append(mini-10)
        elif r > maxi:
            rescaled.append(maxi+10)
        else:
            rescaled.append(r)
    return rescaled
    

# --- Main execution ---
if __name__ == "__main__":
    input_file = "/mnt/data/user/zhao_jun/mou_yu_rong/openrlhf/chessBattleAdvanced/ChessBattleAssessment/evaluation_results_vllm/grpo/3games_4_1600each.jsonl"
    output_file = '/dev/null'#input_file.replace('.jsonl', '_reformatted_rescaled.jsonl')
    
    process_jsonl_file(input_file, output_file)