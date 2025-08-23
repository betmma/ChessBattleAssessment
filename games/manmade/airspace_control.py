import copy
import logging
from typing import List, Tuple, Optional, Any
import numpy as np
from games.board_game import BoardGame


class AirspaceControlGame(BoardGame, board_size=(5, 5), move_arity=2):
    """
    Airspace Control game implementation as a subclass of BoardGame.
    
    A strategic territory control game where players deploy aircraft to control airspace zones.
    Players score points by creating flight paths (connected lines) and controlling landing zones.
    """
    
    # Game introduction for the BoardGame system
    game_introduction = (
        "Airspace Control is played on a 5x5 grid representing airspace zones. "
        "Players take turns placing their aircraft (X or O) in empty cells to control airspace. "
        "Score points by creating 'flight paths' - horizontal, vertical, or diagonal lines of 3+ connected aircraft. "
        "Bonus points for controlling 'landing zones' (corners worth 2 points, center worth 1 point). "
        "Flight paths score: 3-aircraft=3pts, 4-aircraft=5pts, 5-aircraft=8pts. "
        "Game ends when board is full. Winner has the most total points. "
        "Moves are specified as (row, col) coordinates where row and col are 0-4."
    )
    
    def __init__(self):
        """Initialize the Airspace Control game"""
        super().__init__()
        self._game_over_forced_forfeit = False
    
    def _get_legal_moves(self) -> List[Tuple[int, int]]:
        """Return all legal moves as (row, col) coordinate pairs"""
        legal_moves = []
        for row in range(5):
            for col in range(5):
                if self.board[row, col] == 0:
                    legal_moves.append((row, col))
        return legal_moves
    
    def make_move(self, move: Tuple[int, int]) -> bool:
        """Execute a move, return True if move was legal and successful"""
        if not isinstance(move, tuple) or len(move) != 2:
            return False
            
        row, col = move
        
        # Validate coordinates
        if not (0 <= row < 5 and 0 <= col < 5):
            return False
            
        # Check if the cell is empty
        if self.board[row, col] != 0:
            return False
            
        # Make the move
        self.board[row, col] = self.current_player
        
        # Switch players
        self.current_player = -self.current_player
        
        return True
    
    def check_winner(self) -> Optional[Any]:
        """
        Check if there's a winner based on total points.
        Returns winner (1 or -1), 0 for draw, None if game not over.
        """
        if not self.is_game_over():
            return None
            
        player1_score = self._calculate_player_score(1)
        player2_score = self._calculate_player_score(-1)
        
        if player1_score > player2_score:
            return 1
        elif player2_score > player1_score:
            return -1
        else:
            return 0  # Draw
    
    def is_game_over(self) -> bool:
        """Check if the game is over (board full or forced forfeit)"""
        if self._game_over_forced_forfeit:
            return True
        
        # Game over when board is full
        return len(self._get_legal_moves()) == 0
    
    def _calculate_player_score(self, player: int) -> int:
        """Calculate total score for a player"""
        flight_path_score = self._calculate_flight_paths_score(player)
        landing_zone_score = self._calculate_landing_zones_score(player)
        return flight_path_score + landing_zone_score
    
    def _calculate_flight_paths_score(self, player: int) -> int:
        """Calculate score from flight paths (connected lines)"""
        score = 0
        
        # Check all possible lines on the board
        lines = self._get_all_lines()
        
        for line in lines:
            # Count consecutive pieces for this player in the line
            consecutive_counts = []
            current_count = 0
            
            for pos in line:
                row, col = pos
                if self.board[row, col] == player:
                    current_count += 1
                else:
                    if current_count >= 3:
                        consecutive_counts.append(current_count)
                    current_count = 0
            
            # Don't forget the last sequence
            if current_count >= 3:
                consecutive_counts.append(current_count)
            
            # Add scores for each flight path
            for count in consecutive_counts:
                if count == 3:
                    score += 3
                elif count == 4:
                    score += 5
                elif count >= 5:
                    score += 8
        
        return score
    
    def _calculate_landing_zones_score(self, player: int) -> int:
        """Calculate score from controlling landing zones"""
        score = 0
        
        # Corner landing zones (worth 2 points each)
        corners = [(0, 0), (0, 4), (4, 0), (4, 4)]
        for row, col in corners:
            if self.board[row, col] == player:
                score += 2
        
        # Center landing zone (worth 1 point)
        if self.board[2, 2] == player:
            score += 1
        
        return score
    
    def _get_all_lines(self) -> List[List[Tuple[int, int]]]:
        """Get all possible lines (horizontal, vertical, diagonal) on the board"""
        lines = []
        
        # Horizontal lines
        for row in range(5):
            lines.append([(row, col) for col in range(5)])
        
        # Vertical lines
        for col in range(5):
            lines.append([(row, col) for row in range(5)])
        
        # Main diagonal (top-left to bottom-right)
        lines.append([(i, i) for i in range(5)])
        
        # Anti-diagonal (top-right to bottom-left)
        lines.append([(i, 4-i) for i in range(5)])
        
        # Additional diagonals of length 3+ for more strategic depth
        # Upper diagonals
        for start_col in range(1, 3):  # Start from columns 1,2
            diagonal = []
            row, col = 0, start_col
            while row < 5 and col < 5:
                diagonal.append((row, col))
                row += 1
                col += 1
            if len(diagonal) >= 3:
                lines.append(diagonal)
        
        # Lower diagonals
        for start_row in range(1, 3):  # Start from rows 1,2
            diagonal = []
            row, col = start_row, 0
            while row < 5 and col < 5:
                diagonal.append((row, col))
                row += 1
                col += 1
            if len(diagonal) >= 3:
                lines.append(diagonal)
        
        # Anti-diagonals (upper)
        for start_col in range(2, 4):  # Start from columns 2,3
            diagonal = []
            row, col = 0, start_col
            while row < 5 and col >= 0:
                diagonal.append((row, col))
                row += 1
                col -= 1
            if len(diagonal) >= 3:
                lines.append(diagonal)
        
        # Anti-diagonals (lower)
        for start_row in range(1, 3):  # Start from rows 1,2
            diagonal = []
            row, col = start_row, 4
            while row < 5 and col >= 0:
                diagonal.append((row, col))
                row += 1
                col -= 1
            if len(diagonal) >= 3:
                lines.append(diagonal)
        
        return lines
    
    def evaluate_position(self) -> float:
        """
        Evaluate the current board position for minimax agent.
        Positive values favor player 1, negative values favor player -1.
        """
        if self.is_game_over():
            winner = self.check_winner()
            if winner == 1:
                return 1000.0  # Player 1 wins
            elif winner == -1:
                return -1000.0  # Player -1 wins
            else:
                return 0.0  # Draw
        
        # Calculate current score difference
        player1_score = self._calculate_player_score(1)
        player2_score = self._calculate_player_score(-1)
        
        score_diff = player1_score - player2_score
        
        # Add positional bonuses
        positional_value = 0.0
        
        # Bonus for controlling center early
        if self.board[2, 2] == 1:
            positional_value += 1.0
        elif self.board[2, 2] == -1:
            positional_value -= 1.0
        
        # Small bonus for corner control
        corners = [(0, 0), (0, 4), (4, 0), (4, 4)]
        for row, col in corners:
            if self.board[row, col] == 1:
                positional_value += 0.5
            elif self.board[row, col] == -1:
                positional_value -= 0.5
        
        # Bonus for potential flight paths (partial lines)
        potential_bonus = self._evaluate_potential_flight_paths()
        
        return float(score_diff) + positional_value + potential_bonus
    
    def _evaluate_potential_flight_paths(self) -> float:
        """Evaluate potential for future flight paths"""
        potential_value = 0.0
        lines = self._get_all_lines()
        
        for line in lines:
            player1_count = 0
            player2_count = 0
            empty_count = 0
            
            for row, col in line:
                cell = self.board[row, col]
                if cell == 1:
                    player1_count += 1
                elif cell == -1:
                    player2_count += 1
                else:
                    empty_count += 1
            
            # If line has pieces from only one player, it has potential
            if player1_count > 0 and player2_count == 0 and empty_count > 0:
                potential_value += player1_count * 0.1
            elif player2_count > 0 and player1_count == 0 and empty_count > 0:
                potential_value -= player2_count * 0.1
        
        return potential_value
    
    def get_current_player(self) -> Any:
        """Get the current player"""
        return self.current_player
    
    def force_game_over(self):
        """Force the game to end (used in evaluation)"""
        self._game_over_forced_forfeit = True
