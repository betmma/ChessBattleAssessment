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

    game_introduction = None # should be set by subclasses, ensuring it provides sufficient information for playing the game, including all rules, objectives and move format. Do not write docstring as all information should be in the game_introduction.
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

    def reset(self) -> None:
        """Resets the game to its initial state."""
        self.board = self._create_initial_board()
        self.current_player = 1

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
        Note that this function always stands for player 1's perspective.
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
