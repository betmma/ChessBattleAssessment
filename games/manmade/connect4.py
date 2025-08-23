import copy
import logging
import re
from typing import List, Tuple, Optional, Any
import numpy as np
from games.board_game import BoardGame


class Connect4Game(BoardGame, board_size=(6, 7), move_arity=1):
    """Connect 4 game implementation as a subclass of BoardGame"""
    
    # Game introduction for the BoardGame system
    game_introduction = (
        "Connect 4 is played on a 6x7 vertical grid. "
        "Players take turns dropping their pieces (Red 'R' or Yellow 'Y') into columns. "
        "Pieces fall to the lowest available position in the selected column. "
        "The goal is to get four of your pieces in a row horizontally, vertically, or diagonally. "
        "If the board fills up without a winner, the game is a draw. "
        "Moves are specified as a single column number from 0 to 6."
    )
    
    # Override player symbols for Connect4
    player_symbols = {1: 'R', -1: 'Y', 0: '.'}
    
    def __init__(self):
        """Initialize the Connect4 game"""
        super().__init__()
        self._game_over_forced_forfeit = False
        self.rows = 6
        self.cols = 7
    
    def _get_legal_moves(self) -> List[int]:
        """Return all legal moves as column numbers"""
        return [col for col in range(self.cols) if self.board[0, col] == 0]
    
    def make_move(self, move: int) -> bool:
        """Execute a move, return True if move was legal and successful"""
        if not isinstance(move, int) or move < 0 or move >= self.cols:
            return False
        
        # Check if column is full
        if self.board[0, move] != 0:
            return False
            
        # Find the lowest empty row in the column
        for row in range(self.rows - 1, -1, -1):
            if self.board[row, move] == 0:
                self.board[row, move] = self.current_player
                self.current_player *= -1  # Switch player (1 -> -1, -1 -> 1)
                return True
        
        return False
    
    def check_winner(self) -> Optional[int]:
        """Check for a winner, return 1 (Red wins), -1 (Yellow wins), 0 (draw), None (game continues)"""
        # Check for 4 in a row (horizontal, vertical, diagonal)
        directions = [
            (0, 1),   # Horizontal
            (1, 0),   # Vertical
            (1, 1),   # Diagonal \
            (1, -1)   # Diagonal /
        ]
        
        for row in range(self.rows):
            for col in range(self.cols):
                if self.board[row, col] == 0:
                    continue
                    
                player = self.board[row, col]
                
                for dr, dc in directions:
                    count = 1
                    # Check positive direction
                    r, c = row + dr, col + dc
                    while 0 <= r < self.rows and 0 <= c < self.cols and self.board[r, c] == player:
                        count += 1
                        r, c = r + dr, c + dc
                    
                    # Check negative direction
                    r, c = row - dr, col - dc
                    while 0 <= r < self.rows and 0 <= c < self.cols and self.board[r, c] == player:
                        count += 1
                        r, c = r - dr, c - dc
                    
                    if count >= 4:
                        return int(player)
        
        # Check for draw (board full)
        if all(self.board[0, col] != 0 for col in range(self.cols)):
            return 0
        
        return None  # Game continues
    
    def is_game_over(self) -> bool:
        """Check if the game is over"""
        return self.check_winner() is not None or self._game_over_forced_forfeit
    
    def evaluate_position(self) -> float:
        """
        Evaluate the current position from player 1's perspective.
        Uses the same heuristic as the original Connect4 implementation.
        """
        # Check if game is already won
        winner = self.check_winner()
        if winner is not None:
            if winner == 1:
                return 100.0  # Player 1 wins
            elif winner == -1:
                return -100.0  # Player -1 wins
            else:
                return 0.0  # Draw
        
        score = 0.0
        
        # Evaluate all possible 4-in-a-row positions
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]  # horizontal, vertical, diagonal
        
        for row in range(self.rows):
            for col in range(self.cols):
                for dr, dc in directions:
                    window = []
                    for i in range(4):
                        r, c = row + i * dr, col + i * dc
                        if 0 <= r < self.rows and 0 <= c < self.cols:
                            window.append(self.board[r, c])
                        else:
                            break
                    
                    if len(window) == 4:
                        score += self._evaluate_window(window)
        
        return score
    
    def _evaluate_window(self, window):
        """
        Evaluate a 4-piece window for Connect4
        
        Args:
            window: List of 4 pieces (1 for player 1, -1 for player -1, 0 for empty)
            
        Returns:
            Score for this window
        """
        score = 0
        player1_count = window.count(1)
        player2_count = window.count(-1)
        empty_count = window.count(0)
        
        # If both players have pieces in the window, it's blocked
        if player1_count > 0 and player2_count > 0:
            return 0
        
        # Score for player 1 (maximizing player)
        if player1_count == 4:
            score += 100
        elif player1_count == 3 and empty_count == 1:
            score += 10
        elif player1_count == 2 and empty_count == 2:
            score += 2
        
        # Score for player -1 (minimizing player)
        if player2_count == 4:
            score -= 100
        elif player2_count == 3 and empty_count == 1:
            score -= 10
        elif player2_count == 2 and empty_count == 2:
            score -= 2
        
        return score
    
    def get_board_representation_for_llm(self) -> str:
        """Get board state representation for LLM"""
        s = "Current Connect 4 board state:\n"
        s += "Columns: " + " ".join([str(i) for i in range(self.cols)]) + "\n"
        
        for row in range(self.rows):
            row_parts = []
            for col in range(self.cols):
                symbol = self.player_symbols.get(self.board[row, col], str(self.board[row, col]))
                row_parts.append(f"{symbol}")
            s += "         " + " ".join(row_parts) + "\n"
        
        return s.strip()
    
    def parse_move_from_output(self, raw_output: str) -> Optional[int]:
        """
        Parse move from LLM output for Connect 4.
        This overrides the BoardGame method to handle Connect4-specific parsing.
        
        Args:
            raw_output: Raw text output from the LLM
            
        Returns:
            Parsed move as column number or None if no valid move found
        """
        # First try the BoardGame format (tuple parsing)
        move_tuple = super().parse_move_from_output(raw_output)
        if move_tuple is not None and isinstance(move_tuple, tuple) and len(move_tuple) == 1:
            return move_tuple[0]
        
        # Look for column numbers in various formats
        # First try to find numbers in brackets
        bracket_match = re.findall(r'\[(\d+)\]', raw_output)
        if bracket_match:
            try:
                move = int(bracket_match[-1])
                if 0 <= move <= 6:
                    return move
            except ValueError:
                pass
        
        # Then try to find numbers after thinking block or at end
        think_match = re.search(r'</think>\s*(\d+)', raw_output)
        if think_match:
            try:
                move = int(think_match.group(1))
                if 0 <= move <= 6:
                    return move
            except ValueError:
                pass
        
        # Finally, try to find any single digit that's a legal move
        digit_matches = re.findall(r'\b([0-6])\b', raw_output)
        for match in reversed(digit_matches):  # Check from end to beginning
            try:
                move = int(match)
                if 0 <= move <= 6:
                    return move
            except ValueError:
                continue
        
        logging.warning(f"Connect4: Could not parse move from output: '{raw_output}'")
        return None
    
    def reset(self) -> None:
        """Reset game to initial state"""
        super().reset()
        self._game_over_forced_forfeit = False
    
    def force_forfeit(self) -> None:
        """Force game end (for evaluation)"""
        self._game_over_forced_forfeit = True
    
    def clone(self):
        """Create a deep copy of the game"""
        new_game = self.__class__()
        new_game.board = np.copy(self.board)
        new_game.current_player = self.current_player
        new_game._game_over_forced_forfeit = self._game_over_forced_forfeit
        return new_game
    
    def load_state_from_representation(self, state_str: str) -> bool:
        """
        Load Connect4 game state from string representation.
        
        Args:
            state_str: String representation from get_state_representation()
            
        Returns:
            bool: True if state was loaded successfully, False if parsing failed
        """
        try:
            lines = state_str.strip().split('\n')
            
            # Check if this looks like a valid Connect4 state
            if not any("Connect 4 board state" in line for line in lines):
                return False
            
            # Find the board section (starts with "Columns:")
            board_start = -1
            for i, line in enumerate(lines):
                if line.strip().startswith("Columns:"):
                    board_start = i + 1
                    break
            
            if board_start == -1:
                return False
            
            # Parse board rows
            new_board = np.zeros((self.rows, self.cols), dtype=int)
            board_row = 0
            
            for i in range(board_start, len(lines)):
                line = lines[i].strip()
                if line.startswith("Current turn:"):
                    break
                
                # Skip empty lines and lines that don't look like board rows
                if not line or not any(c in line for c in ['R', 'Y', '.']):
                    continue
                
                # Extract symbols from the line
                symbols = []
                for char in line.split():
                    if char in ['R', 'Y', '.']:
                        symbols.append(char)
                
                if len(symbols) == self.cols and board_row < self.rows:
                    for col in range(self.cols):
                        if symbols[col] == 'R':
                            new_board[board_row, col] = 1
                        elif symbols[col] == 'Y':
                            new_board[board_row, col] = -1
                        else:  # '.'
                            new_board[board_row, col] = 0
                    board_row += 1
            
            # Check if we parsed the correct number of rows
            if board_row != self.rows:
                return False
            
            # Parse current player
            current_player = 1
            found_current_turn = False
            for line in lines:
                if line.strip().startswith("Current turn:"):
                    found_current_turn = True
                    if "(plays as -1)" in line:
                        current_player = -1
                    elif "(plays as 1)" in line:
                        current_player = 1
                    break
            
            if not found_current_turn:
                return False
            
            # Update game state
            self.board = new_board
            self.current_player = current_player
            self._game_over_forced_forfeit = False
            
            return True
            
        except Exception:
            return False
