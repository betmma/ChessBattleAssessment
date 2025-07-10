from abc import abstractmethod
from typing import List, Any, Optional, Tuple
import numpy as np
import re
from games.game import Game

class BoardGame(Game):
    """
    An abstract base class for board games, inheriting from the base Game class.
    It handles games with a 1D or 2D array-based board and n-ary tuple moves.
    """

    game_introduction = None # should be set by subclasses, ensuring it provides sufficient information for playing the game, including rules, objectives and move format.
    system_prompt = (
        "You are an expert in the {game_name} game.\n"
        "{game_introduction}\n"
        "After thinking, you should respond with a move in the format: {move_format}.\n"
    )
    system_prompt_no_thinking = (
        "You are an expert in the {game_name} game.\n"
        "{game_introduction}\n"
        "You must respond with a move in the format: {move_format} and nothing else.\n"
    )
    user_prompt_template = user_prompt_template_no_thinking = (
        "{board_representation}\n"
        "You are player '{player_symbol}'.\n"
        "Your available legal moves: [{legal_moves_str}]\n"
    )
    board_size: Tuple[int, ...]
    move_arity: int
    player_symbols = {1: 'X', -1: 'O', 0: '.'} # Player symbols, default use is to form user prompt as seen above.

    def __init_subclass__(cls, board_size: Tuple[int, ...], move_arity: int, **kwargs):
        """
        Initializes the subclass with board dimensions and move arity.

        Args:
            board_size: A tuple defining the dimensions of the board.
            move_arity: An integer defining the number of elements in a move tuple.
        """
        super().__init_subclass__(**kwargs)
        cls.board_size = board_size
        cls.move_arity = move_arity
        if not hasattr(cls, 'game_introduction'):
            raise ValueError(
                f"Subclasses of BoardGame must define a 'game_introduction' class attribute. "
                f"Class {cls.__name__} does not have it defined."
            )
        move_format= f"({', '.join([chr(97 + i) for i in range(move_arity)])})"
        cls.system_prompt = cls.system_prompt.format(game_name=cls.name, game_introduction=cls.game_introduction, move_format=move_format)
        cls.system_prompt_no_thinking = cls.system_prompt_no_thinking.format(game_name=cls.name, game_introduction=cls.game_introduction, move_format=move_format)

    def __init__(self):
        """
        Initializes the board game using class-level board_size.
        """
        super().__init__()
        self.current_player = 1
        self.board = self._create_initial_board()
    
    def get_board_representation_for_llm(self) -> str:
        """Returns a string representation of the board for the LLM."""
        if self.board.ndim == 1:
            # 1D board representation
            return " ".join([self.player_symbols.get(cell, str(cell)) for cell in self.board])
        return "\n".join([" ".join([self.player_symbols.get(cell, str(cell)) for cell in row]) for row in self.board])

    def get_key_for_cache(self) -> tuple:
        """Returns a cache key for the current board state."""
        return (self.board.tobytes(),)

    def load_state_from_representation(self, state_str: str) -> bool:
        """
        Loads the game state from a string representation, assuming the format
        from the base Game.get_state_representation method.
        """
        try:
            parts = state_str.strip().split('\n')
            
            # Determine where the board representation ends
            board_lines = []
            i = 0
            for i, line in enumerate(parts):
                if line.startswith("Current turn:"):
                    break
                board_lines.append(line)
            else: # if no break
                i = len(parts)

            board_repr_str = "\n".join(board_lines)
            
            # Reconstruct board from string
            board_rows = [row.split() for row in board_repr_str.split('\n')]
            
            # Validate dimensions
            if len(self.board_size) == 2:
                rows, cols = self.board_size
                if len(board_rows) != rows or not all(len(r) == cols for r in board_rows):
                    return False
            elif len(self.board_size) == 1:
                if len(board_rows) != 1 or len(board_rows[0]) != self.board_size[0]:
                    return False
            else: # Not 1D or 2D
                return False

            symbol_to_player = {v: k for k, v in self.player_symbols.items()}
            self.board = np.array([[symbol_to_player.get(s, 0) for s in row] for row in board_rows], dtype=int).reshape(self.board_size)

            # Find and parse the current player line
            player_line_found = False
            for line in parts[i:]:
                match = re.search(r"Current turn: . \(plays as (-?\d+)\)", line)
                if match:
                    self.current_player = int(match.group(1))
                    player_line_found = True
                    break
            
            return player_line_found
        except Exception:
            return False

    def _get_player_symbol(self, player_value: Any) -> str:
        """Returns the symbol for a given player."""
        return self.player_symbols.get(player_value, '?')

    def parse_move_from_output(self, raw_output: str) -> Optional[Any]:
        """Parses a move from the raw output of an agent, enforcing move arity."""
        # Regex to find tuples of numbers, e.g., (1, 2) or (3)
        pattern = r'\((' + r'\s*\d+\s*,' * (self.move_arity - 1) + r'\s*\d+\s*' + r')\)' if self.move_arity > 1 else r'\(\s*\d+\s*\)'
        match = None
        for next_match in re.finditer(pattern, raw_output):
            match = next_match
        
        if match:
            try:
                # Safely evaluate the matched string to a tuple
                move_tuple = eval(match.group(0))
                
                # For single-element tuples, eval might produce just a number
                if isinstance(move_tuple, int) and self.move_arity == 1:
                    move_tuple = (move_tuple,)

                if isinstance(move_tuple, tuple) and len(move_tuple) == self.move_arity:
                    return move_tuple
            except:
                return None
        return None

    def reset(self) -> None:
        """Resets the game to its initial state."""
        self.board = self._create_initial_board()
        self.current_player = 1

    def clone(self):
        """Creates a deep copy of the game state."""
        new_game = self.__class__()
        new_game.board = np.copy(self.board)
        new_game.current_player = self.current_player
        return new_game

    # ----------------------------------------------------------------
    # methods can be overridden by specific board game classes
    # ----------------------------------------------------------------
    
    def _create_initial_board(self) -> np.ndarray:
        """
        Creates and returns the initial board configuration.
        This method is called by __init__ and reset.
        """
        return np.zeros(self.board_size, dtype=int)

    def evaluate_position(self) -> float:
        """
        Evaluates the current board position for minimax agent. Default implementation returns 0.
        This should be overridden by subclasses with game-specific evaluation logic.
        Positive values favor player 1, negative values favor player -1.
        """
        return 0.0

    def get_legal_moves(self) -> List[Any]:
        """Return all currently legal moves, with numpy types cleaned."""
        from utils.safe_json_dump import clean_np_types
        moves = self._get_legal_moves()
        return clean_np_types(moves)

    # ----------------------------------------------------------------
    # Abstract methods to be implemented by specific board game classes
    # ----------------------------------------------------------------

    @abstractmethod
    def _get_legal_moves(self) -> List[Any]:
        """Return all currently legal moves."""
        pass

    @abstractmethod
    def make_move(self, move: Any) -> bool:
        """Executes a move."""
        pass

    @abstractmethod
    def check_winner(self) -> Optional[Any]:
        """Checks for a winner."""
        pass

    @abstractmethod
    def is_game_over(self) -> bool:
        """Checks if the game is over."""
        pass

    def get_current_player(self) -> Any:
        """Gets the current player."""
        return self.current_player
