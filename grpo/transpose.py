import json

def transpose_connect4_board_in_content(content_str: str) -> str:
    """
    Finds a Connect 4 board in the content string, transposes it,
    and returns the modified content string.
    """
    lines = content_str.splitlines()
    new_content_lines = []
    
    board_data_rows = 6  # Standard Connect 4 height
    board_columns_header_prefix = "Columns:"
    
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped_line = line.strip()

        if stripped_line.startswith(board_columns_header_prefix):
            # Found the board header. The next `board_data_rows` are the board
            new_content_lines.append('Each column is in the format: Column <index>: <bottom-to-top cell values>')
            
            current_board_rows_str = []
            for j in range(1, board_data_rows + 1):
                if i + j < len(lines):
                    current_board_rows_str.append(lines[i + j])
                else:
                    # Should not happen with valid input, but good to be safe
                    print(f"Warning: Board seems truncated. Expected {board_data_rows} rows after header.")
                    # Add original lines and stop processing this board
                    new_content_lines.append(line) 
                    new_content_lines.extend(current_board_rows_str)
                    i += len(current_board_rows_str) + 1
                    continue # continue outer while loop


            # Parse the board into a 2D list (grid[row][col])
            grid = []
            for r_str in current_board_rows_str:
                # Cells are space-separated. Need to handle potential leading/trailing spaces on the line.
                cells = r_str.strip().split(' ')
                grid.append(cells)

            if not grid or not grid[0]:
                # Malformed board, add original lines and skip
                print("Warning: Malformed or empty board data found.")
                new_content_lines.append(line)
                new_content_lines.extend(current_board_rows_str)
                i += board_data_rows + 1 
                continue

            num_actual_rows = len(grid)
            num_cols = len(grid[0])

            # Transpose and format
            transposed_board_lines = []
            for c in range(num_cols):
                column_data_top_to_bottom = []
                for r in range(num_actual_rows):
                    if c < len(grid[r]): # Ensure column index is valid for this row
                        column_data_top_to_bottom.append(grid[r][c])
                    else:
                        # Should not happen if board is rectangular
                        column_data_top_to_bottom.append('?') # Placeholder for error

                column_data_bottom_to_top = column_data_top_to_bottom[::-1] # Reverse for bottom-to-top
                transposed_board_lines.append(f"Column {c}: {' '.join(column_data_bottom_to_top)}")
            
            new_content_lines.extend(transposed_board_lines)
            i += (1 + board_data_rows) # Move past the original header and board rows
        else:
            if not stripped_line.startswith('Your available legal moves (columns): '): # remove legal moves line
                new_content_lines.append(line)
            i += 1
            
    return "\n".join(new_content_lines)

def process_jsonl_file(input_filepath: str, output_filepath: str):
    """
    Reads a JSONL file, processes the Connect 4 board in specified fields,
    and writes the modified data to a new JSONL file.
    """
    with open(input_filepath, 'r', encoding='utf-8') as infile, \
         open(output_filepath, 'w', encoding='utf-8') as outfile:
        
        for line_num, line_str in enumerate(infile):
            try:
                data = json.loads(line_str)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON on line {line_num + 1}: {e}")
                outfile.write(line_str) # Write original line if it's broken
                continue

            if 999 not in data['reward_model']["ground_truth"]: # not win in 1
                continue
            if isinstance(data.get("prompt"), list) and len(data["prompt"]) > 1:
                if isinstance(data["prompt"][1], dict) and "content" in data["prompt"][1]:
                    original_content = data["prompt"][1]["content"]
                    modified_content = transpose_connect4_board_in_content(original_content)
                    data["prompt"][1]["content"] = modified_content
            
            outfile.write(json.dumps(data) + '\n')

    print(f"Processing complete. Output written to {output_filepath}")

# --- Main execution ---
if __name__ == "__main__":
    input_file = "/mnt/data/user/zhao_jun/mou_yu_rong/openrlhf/chessBattleAdvanced/ChessBattleAssessment/evaluation_results_vllm/grpo/grpo8.jsonl"
    output_file = input_file.replace('.jsonl', '_win_in_1.jsonl')
    process_jsonl_file(input_file, output_file)