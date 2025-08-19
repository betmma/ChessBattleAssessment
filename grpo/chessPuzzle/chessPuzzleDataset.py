import pandas as pd
import chess
import random
import re
import threading
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ChessPuzzleDataset:
    """
    A dynamic dataset for chess puzzles that adapts to a model's Elo rating.

    This class loads the Lichess puzzle database, tracks a model's Elo, and serves
    puzzles with a difficulty rating close to the model's current Elo. It also
    provides a method to update the Elo based on puzzle results (win/loss).
    """

    def __init__(self, csv_path: str, initial_model_elo: int = 1500, k_factor: int = 32, elo_range: int = 100):
        """
        Args:
            csv_zst_path (str): Path to the lichess_db_puzzle.csv.zst file.
            initial_model_elo (int): The starting Elo for the model.
            k_factor (int): The K-factor for Elo calculation. Determines how much the Elo changes.
            elo_range (int): The range (+/-) around the model's Elo from which to select puzzles.
        """
        self.csv_path = csv_path
        self.model_elo = initial_model_elo
        self.k_factor = k_factor
        self.elo_range = elo_range
        self.elo_update_lock = threading.Lock() # For thread-safe Elo updates in a real training env
        # Cache to ensure the same index always returns the same puzzle/sample during the dataset's lifetime
        self._index_cache = {}
        self._cache_lock = threading.Lock()

        # Preprocess data
        self._load_and_preprocess_data()
        
    def __len__(self):
        """Returns the number of puzzles available in the dataset."""
        return len(self.puzzles_df)

    def _load_and_preprocess_data(self):
        """Loads and prepares the puzzle data for efficient access."""
        logging.info("Loading and preprocessing puzzle database...")
        # Use pandas to read the compressed CSV
        df = pd.read_csv(self.csv_path)

        is_one_step_puzzle = df['Moves'].str.split(' ').str.len() == 2
        df = df[is_one_step_puzzle].copy()
        # The first move in 'Moves' is the opponent's move. The second is the solution.
        # We need both to set up the puzzle correctly.
        moves_split = df['Moves'].str.split(' ', n=2, expand=True)
        df['OpponentMove'] = moves_split[0]
        df['SolutionMove'] = moves_split[1].str.split(' ', n=1, expand=True)[0]

        # The FEN in the CSV is before the opponent's move.
        # We need the FEN for the player to solve.
        # This is computationally expensive to do for all 5M+ puzzles upfront.
        # We will do it dynamically when a puzzle is selected.
        self.puzzles_df = df[['PuzzleId', 'FEN', 'OpponentMove', 'SolutionMove', 'Rating']].copy()
        
        # For fast lookup, bin puzzles by their rating
        logging.info("Binning puzzles by rating for efficient lookup...")
        rating_step = 50
        self.puzzles_df['RatingBin'] = (self.puzzles_df['Rating'] // rating_step) * rating_step
        self.binned_puzzles = self.puzzles_df.groupby('RatingBin').groups
        self.available_bins = sorted(self.binned_puzzles.keys())
        logging.info(f"Preprocessing complete. Loaded {len(self.puzzles_df)} puzzles.")

    def _get_player_fen(self, start_fen: str, opponent_move_uci: str) -> str:
        """Applies the opponent's move to the FEN to get the position for the player."""
        try:
            board = chess.Board(start_fen)
            move = chess.Move.from_uci(opponent_move_uci)
            if move in board.legal_moves:
                board.push(move)
                return board.fen()
            else:
                # This should be rare with the Lichess dataset but is good practice
                logging.warning(f"Illegal opponent move '{opponent_move_uci}' for FEN '{start_fen}'.")
                return None
        except Exception as e:
            logging.error(f"Error processing move '{opponent_move_uci}' on FEN '{start_fen}': {e}")
            return None

    def _fen_to_text(self, fen: str) -> str:
        """
        Converts a FEN string to a human-readable text description of the board.
        """
        try:
            board = chess.Board(fen)
        except ValueError:
            return "Invalid FEN string." # Graceful handling of bad FENs

        piece_map = {
            'p': 'black pawn', 'r': 'black rook', 'n': 'black knight',
            'b': 'black bishop', 'q': 'black queen', 'k': 'black king',
            'P': 'white pawn', 'R': 'white rook', 'N': 'white knight',
            'B': 'white bishop', 'Q': 'white queen', 'K': 'white king',
        }
        
        description_parts = []
        for square_index in range(64):
            square_name = chess.square_name(square_index)
            piece = board.piece_at(square_index)
            
            if piece:
                piece_text = piece_map[piece.symbol()]
                description_parts.append(f"{square_name}: {piece_text}")
            # Optional: You could add "empty" for clarity, but it might make the prompt too long.
            # else:
            #     description_parts.append(f"{square_name}: empty")

        # Add side to move and castling rights for full context
        side_to_move = "White to move." if board.turn == chess.WHITE else "Black to move."
        castling_rights = board.castling_xfen()
        castling_description = {
            'K': 'White can castle kingside',
            'Q': 'White can castle queenside',
            'k': 'Black can castle kingside',
            'q': 'Black can castle queenside'
        }
        castling_rights = ', '.join([desc for key, desc in castling_description.items() if key in castling_rights]) or "No castling rights available."
        
        description = "Board state:\n" + "\n".join(description_parts)
        description += f"\n\n{side_to_move}"
        description += f"\nCastling rights: {castling_rights}"

        return description
    
    def __getitem__(self, index):
        """Yields puzzle samples formatted for the training program."""
        # If we've already generated a sample for this index, return it to ensure determinism per index.
        with self._cache_lock:
            cached = self._index_cache.get(index)
        if cached is not None:
            return cached

        while True:
            # 1. Select a puzzle based on the current model Elo
            target_elo_bin = (int(self.model_elo) // 50) * 50
            
            # Find a suitable bin, falling back if the exact one is empty
            min_bin = max(self.available_bins[0], target_elo_bin - self.elo_range)
            max_bin = min(self.available_bins[-1], target_elo_bin + self.elo_range)
            
            possible_bins = [b for b in self.available_bins if min_bin <= b <= max_bin]
            if not possible_bins:
                # Fallback if no bins in range (e.g., very high/low Elo)
                possible_bins = [min(self.available_bins, key=lambda b: abs(b - target_elo_bin))]

            selected_bin = random.choice(possible_bins)
            puzzle_indices = self.binned_puzzles[selected_bin]
            puzzle_idx = random.choice(puzzle_indices)
            
            puzzle = self.puzzles_df.loc[puzzle_idx]

            # 2. Prepare the puzzle for the model
            player_fen = self._get_player_fen(puzzle['FEN'], puzzle['OpponentMove'])
            if player_fen is None:
                continue # Skip malformed puzzles

            board_description = self._fen_to_text(player_fen)
            # 3. Format the sample as specified
            prompt = [
                {"role": "system", "content": "You are a chess puzzle solver. Analyze the position and provide the best move in algebraic notation (SAN). Examples: Nf3, Qxe5+, O-O, e8=Q#. Reply with only the move."},
                {"role": "user", "content": f"Solve the following chess puzzle (answer in SAN): {board_description}"}
            ]
            
            # Pack ground truth info for the reward function
            reward_model_data = {
                "ground_truth": {
                    "solution_move": puzzle['SolutionMove'],
                    "puzzle_elo": puzzle['Rating'],
                    "player_fen": player_fen,
                }
            }
            
            sample = {"prompt": prompt, "reward_model": reward_model_data}
            with self._cache_lock:
                self._index_cache[index] = sample
            return sample

    def update_elo(self, puzzle_elo: int, score: float):
        """
        Updates the model's Elo rating based on the outcome of a puzzle.
        
        Args:
            puzzle_elo (int): The Elo rating of the puzzle that was attempted.
            score (float): The result for the model (1.0 for a win, 0.0 for a loss, 0.5 for a draw).
        """
        with self.elo_update_lock:
            expected_score = 1 / (1 + 10 ** ((puzzle_elo - self.model_elo) / 400))
            new_elo = self.model_elo + self.k_factor * (score - expected_score)
            logging.info(
                f"Elo Update: Current Elo={self.model_elo:.0f}, Puzzle Elo={puzzle_elo}, "
                f"Score={score}, Expected={expected_score:.2f}, New Elo={new_elo:.0f}"
            )
            self.model_elo = new_elo