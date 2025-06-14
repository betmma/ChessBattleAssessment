import os
import sys
import logging
import argparse
from typing import Optional

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from config import Config, setup_logging
from games import Game, TicTacToeGame, Connect4Game
from agents import RandomAgent, MinimaxAgent, APIAgent, VLLMAgent
from utils import create_agent


class HumanPlayer:
    """Human player that gets input from console"""
    
    def __init__(self, name="Human"):
        self.name = name
    
    def get_move(self, game):
        """Get move from human player via console input"""
        legal_moves = game.get_legal_moves()
        
        print(f"\n{self.name}, it's your turn!")
        print(f"Legal moves: {legal_moves}")
        
        while True:
            try:
                move_input = input("Enter your move (format depends on game): ").strip()
                
                # Try to parse the move
                parsed_move = game.parse_move_from_output(move_input, legal_moves)
                
                if parsed_move is not None and parsed_move in legal_moves:
                    return str(parsed_move)
                else:
                    print(f"Invalid move '{move_input}'. Please enter a valid move from: {legal_moves}")
                    
            except KeyboardInterrupt:
                print("\nGame interrupted by user.")
                sys.exit(0)
            except Exception as e:
                print(f"Error parsing move: {e}. Please try again.")



def parse_args():
    parser = argparse.ArgumentParser(description="Play against an AI agent interactively")
    
    # Game configuration
    parser.add_argument("--game", type=str, default="tictactoe",
                        choices=["tictactoe", "connect4"],
                        help="Game to play")
    
    # Agent configuration
    parser.add_argument("--agent", type=str, required=True, 
                        choices=["api", "random", "minimax", "vllm"],
                        help="AI agent type to play against")
    parser.add_argument("--agent_model", type=str, default="gpt-4-0125-preview", 
                        help="Model name for agent (only used if agent is 'api')")
    parser.add_argument("--agent_api_base_url", type=str, 
                        help="API base URL for agent (required if agent is 'api')")
    parser.add_argument("--agent_api_key", type=str, 
                        help="API key for agent (required if agent is 'api')")
    parser.add_argument("--agent_model_path", type=str, 
                        default='Qwen/Qwen3-8B',
                        help="Model path for agent (only used if agent is 'vllm')")
    
    # Game options
    parser.add_argument("--human_first", action="store_true",
                        help="Human player goes first (default: AI goes first)")
    parser.add_argument("--human_name", type=str, default="Human",
                        help="Name for human player")
    
    return parser.parse_args()


def play_game(game_class, human_player, ai_agent, human_first=False):
    """Play a single game between human and AI"""
    game:Game = game_class()
    
    # Determine player order
    if human_first:
        player1, player2 = human_player, ai_agent
        print(f"\n{human_player.name} plays first (X), {ai_agent.name} plays second (O)")
    else:
        player1, player2 = ai_agent, human_player
        print(f"\n{ai_agent.name} plays first (X), {human_player.name} plays second (O)")
    
    current_player = player1
    move_count = 0
    
    while not game.is_game_over():
        print(f"\n--- Move {move_count + 1} ---")
        print(game.get_state_representation())
        
        try:
            if isinstance(current_player, HumanPlayer):
                move = current_player.get_move(game)
            else:
                print(f"\n{current_player.name} is thinking...")
                move = current_player.get_move(game)
                print(f"{current_player.name} chooses: {move}")
            
            # Parse and apply move
            legal_moves = game.get_legal_moves()
            parsed_move = game.parse_move_from_output(str(move), legal_moves)
            
            if parsed_move is None or parsed_move not in legal_moves:
                print(f"Invalid move from {current_player.name}: {move}")
                if isinstance(current_player, HumanPlayer):
                    continue
                else:
                    print("AI made invalid move - game ends")
                    break
            
            # Apply the move
            success = game.make_move(parsed_move)
            if not success:
                print(f"Failed to apply move: {move}")
                break
            
            # Switch players
            current_player = player2 if current_player == player1 else player1
            move_count += 1
            
        except KeyboardInterrupt:
            print("\nGame interrupted by user.")
            return None
        except Exception as e:
            print(f"Error during game: {e}")
            return None
    
    # Game over - show final state
    print(f"\n--- GAME OVER ---")
    print(game.get_state_representation())
    
    # Determine winner
    winner = game.check_winner()
    if winner is None:
        print("It's a draw!")
        return "draw"
    elif winner == "X":
        winner_name = player1.name
        print(f"{winner_name} wins!")
        return player1
    else:  # winner == "O"
        winner_name = player2.name
        print(f"{winner_name} wins!")
        return player2


def main():
    args = parse_args()
    
    # Initialize config and logging
    config = Config()
    logger = setup_logging(config)
    
    print("=" * 50)
    print("INTERACTIVE GAME: HUMAN VS AI")
    print("=" * 50)
    
    # Create human player
    human_player = HumanPlayer(name=args.human_name)
    
    # Create AI agent
    try:
        ai_agent = create_agent(
            args.agent, 
            f"AI-{args.agent}",
            model=args.agent_model,
            api_base_url=args.agent_api_base_url,
            api_key=args.agent_api_key,
            model_path=args.agent_model_path
        )
        print(f"Successfully initialized AI agent: {ai_agent.name}")
    except Exception as e:
        logger.critical(f"Failed to initialize AI agent ({args.agent}): {e}")
        return 1
    
    # Select game
    if args.game == "tictactoe":
        game_class = TicTacToeGame
        print("Game: Tic Tac Toe")
    elif args.game == "connect4":
        game_class = Connect4Game
        print("Game: Connect 4")
    else:
        print(f"Unknown game: {args.game}")
        return 1
    
    print(f"Players: {human_player.name} vs {ai_agent.name}")
    
    # Game loop
    wins_human = 0
    wins_ai = 0
    draws = 0
    game_number = 1
    
    while True:
        print(f"\n{'='*20} GAME {game_number} {'='*20}")
        
        try:
            # Play one game
            result = play_game(game_class, human_player, ai_agent, args.human_first)
            
            if result == human_player:
                wins_human += 1
            elif result == ai_agent:
                wins_ai += 1
            elif result == "draw":
                draws += 1
            
            # Show current score
            print(f"\n--- SCORE ---")
            print(f"{human_player.name}: {wins_human}")
            print(f"{ai_agent.name}: {wins_ai}")
            print(f"Draws: {draws}")
            
            # Ask if user wants to play again
            while True:
                play_again = input("\nPlay another game? (y/n): ").strip().lower()
                if play_again in ['y', 'yes']:
                    game_number += 1
                    # Alternate who goes first each game
                    args.human_first = not args.human_first
                    break
                elif play_again in ['n', 'no']:
                    print("\nThanks for playing!")
                    print(f"\nFinal Score:")
                    print(f"{human_player.name}: {wins_human}")
                    print(f"{ai_agent.name}: {wins_ai}")
                    print(f"Draws: {draws}")
                    return 0
                else:
                    print("Please enter 'y' or 'n'")
                    
        except KeyboardInterrupt:
            print("\nGoodbye!")
            return 0
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return 1


if __name__ == "__main__":
    sys.exit(main())