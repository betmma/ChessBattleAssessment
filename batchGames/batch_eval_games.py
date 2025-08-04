import sys,os,pkgutil,json
from collections import Counter
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import Config, setup_logging
import logging
from enum import Enum
from dataclasses import dataclass
import signal
from contextlib import contextmanager
import multiprocessing as mp
from multiprocessing import Pool
import traceback

from evaluate_board_game import BoardGameBalanceEvaluator
folder = '/root/myr/genGames/0803/games'
output_folder = '/root/myr/genGames/0803/eval'
success_folder = '/root/myr/genGames/0803/successGames'
sys.path.insert(0, folder)
gameClasses = []

finalResults={}
class BasicResultStatus(Enum):
    FAILED_TO_IMPORT = "Failed to import"
    FAILED_TO_EVALUATE = "Failed to evaluate"
    TIME_LIMIT_EXCEEDED = "Time limit exceeded for evaluation"
    ALL_FORFEIT = "All games are forfeited"
    ANY_FORFEIT = "Some games are forfeited"
    ALL_SAME = "All games are the same (unique_states=1)"
    ALL_DRAW = "All games are draws"
    ALL_FIRST_PLAYER_WIN = "All games are won by first player"
    ALL_SECOND_PLAYER_WIN = "All games are won by second player"
    NO_FIRST_PLAYER_WIN = "First player never wins"
    NO_SECOND_PLAYER_WIN = "Second player never wins"
    QUICK_END_IN_4 = "Under some agent pairs, all games end in 4 steps"
    SUCCESS = "Success"


class TimeoutError(Exception):
    pass

@contextmanager
def timeout(seconds):
    """Context manager for setting a timeout using signal alarm"""
    def timeout_handler(signum, frame):
        raise TimeoutError("Operation timed out")
    
    # Set the signal handler and alarm
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)
    
    try:
        yield
    finally:
        # Reset the alarm and handler
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def evaluate_single_game(game_class_info):
    """
    Evaluate a single game class. This function will be run in parallel.
    
    Args:
        game_class_info: tuple of (game_class, folder, output_folder, success_folder, TIME_LIMIT)
    
    Returns:
        tuple: (game_name, status, results_data, summary_data)
    """
    game_class, folder, output_folder, success_folder, TIME_LIMIT = game_class_info
    game_name = game_class.__name__
    
    # Add process-level timeout protection
    import time
    process_start = time.time()
    
    try:
        # Create evaluator in each process to avoid sharing issues
        config = Config()
        config.LOG_ACTION_REWARDS = False
        config.LOG_LEVEL = logging.FATAL
        evaluator = BoardGameBalanceEvaluator(config=config, num_games=30, depth=3)
        
        with timeout(TIME_LIMIT):
            results = evaluator.evaluate_all_metrics(game_class)

        summary = evaluator.format_summary(results)
        output_path = evaluator.save_results(results, os.path.join(output_folder, f"{game_name}_evaluation.json"))
        
        # Analyze results to determine status
        metrics = results['agent_pair_metrics']
        
        def isForfeitAny(agent_pair_metric):
            result = agent_pair_metric['detailed_results']
            return result['forfeits_agent1'] + result['forfeits_agent2'] > 0
        
        def isForfeitAll(agent_pair_metric):
            result = agent_pair_metric['detailed_results']
            return result['forfeits_agent1'] + result['forfeits_agent2'] == result['total_games']
        
        def isDrawAll(agent_pair_metric):
            result = agent_pair_metric['detailed_results']
            return result['draws'] == result['total_games']
        
        def isFirstPlayerWinAll(agent_pair_metric):
            balance = agent_pair_metric['balance']
            result = agent_pair_metric['detailed_results']
            return balance['first_player_wins'] == result['total_games']
        
        def isSecondPlayerWinAll(agent_pair_metric):
            balance = agent_pair_metric['balance']
            result = agent_pair_metric['detailed_results']
            return balance['second_player_wins'] == result['total_games']
        
        def isNoFirstPlayerWin(agent_pair_metric):
            balance = agent_pair_metric['balance']
            result = agent_pair_metric['detailed_results']
            return balance['first_player_wins'] == 0 and result['total_games'] > 0
        
        def isNoSecondPlayerWin(agent_pair_metric):
            balance = agent_pair_metric['balance']
            result = agent_pair_metric['detailed_results']
            return balance['second_player_wins'] == 0 and result['total_games'] > 0
        
        def isAllSame(agent_pair_metric):
            return agent_pair_metric["variation"]['unique_states'] == 1
        
        def length(agent_pair_metric):
            return agent_pair_metric["length"]['average_length']
        
        # Determine status based on metrics
        if all(isForfeitAll(metric) for metric in metrics.values()):
            status = BasicResultStatus.ALL_FORFEIT.value
        elif any(isForfeitAny(metric) for metric in metrics.values()):
            status = BasicResultStatus.ANY_FORFEIT.value
        elif all(isDrawAll(metric) for metric in metrics.values()):
            status = BasicResultStatus.ALL_DRAW.value
        elif all(isFirstPlayerWinAll(metric) for metric in metrics.values()):
            status = BasicResultStatus.ALL_FIRST_PLAYER_WIN.value
        elif all(isSecondPlayerWinAll(metric) for metric in metrics.values()):
            status = BasicResultStatus.ALL_SECOND_PLAYER_WIN.value
        elif all(isAllSame(metric) for metric in metrics.values()):
            status = BasicResultStatus.ALL_SAME.value
        elif all(isNoFirstPlayerWin(metric) for metric in metrics.values()):
            status = BasicResultStatus.NO_FIRST_PLAYER_WIN.value
        elif all(isNoSecondPlayerWin(metric) for metric in metrics.values()):
            status = BasicResultStatus.NO_SECOND_PLAYER_WIN.value
        elif any(length(metric) <= 4 for metric in metrics.values()):
            status = BasicResultStatus.QUICK_END_IN_4.value
        else:
            status = BasicResultStatus.SUCCESS.value
            # Save the game class source code to success folder
            with open(os.path.join(folder, f"{game_name}.py"), 'r') as f:
                game_code = f.read()
            with open(os.path.join(success_folder, f"{game_name}.py"), 'w') as f:
                f.write(game_code)
        
        return (game_name, status, results, summary)
        
    except TimeoutError:
        status = BasicResultStatus.TIME_LIMIT_EXCEEDED.value
        elapsed = time.time() - process_start
        print(f"Timeout exceeded for {game_name}: evaluation took {elapsed:.1f} seconds (limit: {TIME_LIMIT}s)")
        return (game_name, status, None, None)
    except Exception as e:
        status = BasicResultStatus.FAILED_TO_EVALUATE.value
        elapsed = time.time() - process_start
        print(f"Error evaluating {game_name} after {elapsed:.1f}s: {e}")
        traceback.print_exc()
        return (game_name, status, None, None)


for _, module_name, _ in pkgutil.iter_modules([folder]):
    gameClassName = module_name
    try:
        module = __import__(f'{gameClassName}', fromlist=[gameClassName])
        gameClass = getattr(module, gameClassName)
        gameClasses.append(gameClass)
    except Exception as e:
        print(f"Error importing {gameClassName}: {e}")
        finalResults[gameClassName] = BasicResultStatus.FAILED_TO_IMPORT.value

config = Config()
config.LOG_ACTION_REWARDS=False
config.LOG_LEVEL= logging.FATAL
# evaluator = BoardGameBalanceEvaluator(config=config, num_games=30, depth=3)  # Moved to worker function
all_results = []
all_summaries = []
from datetime import datetime
TIME_LIMIT = 300 

# Prepare arguments for multiprocessing
game_args = [(game_class, folder, output_folder, success_folder, TIME_LIMIT) 
             for game_class in gameClasses]

# Determine number of processes (use fewer processes to avoid hanging)
num_processes = max(1, min(127, len(gameClasses)))  # Limit to 2 processes max
print(f"Using {num_processes} processes for parallel evaluation")

# Use multiprocessing to evaluate games in parallel
if __name__ == "__main__":
    import time
    mp.set_start_method('spawn', force=True)  # Ensure compatibility
    
    start_time = time.time()
    print(f"Starting evaluation of {len(gameClasses)} games...")
    
    # Use imap_unordered with timeout for better control
    with Pool(processes=num_processes) as pool:
        try:
            # Use imap_unordered to get results as they complete
            results_iter = pool.imap_unordered(evaluate_single_game, game_args)
            
            # Collect results with a global timeout
            results_list = []
            completed = 0
            
            for result in results_iter:
                results_list.append(result)
                completed += 1
                elapsed = time.time() - start_time
                game_name = result[0]
                status = result[1]
                text=f"[{completed}/{len(gameClasses)}] Completed {game_name}: {status} (elapsed: {elapsed:.1f}s)"
                with open(os.path.join(output_folder, "progress.txt"), 'a') as f:
                    f.write(text + '\n')
                    print(text)
                
                distribution = Counter([r[1] for r in results_list])
                print("\nDistribution of results:")
                for status, count in distribution.items():
                    print(f"{status}: {count}")
                
        except Exception as e:
            print(f"Error during multiprocessing: {e}")
            # Try to terminate the pool gracefully
            pool.terminate()
            pool.join()
            results_list = []
    
    total_time = time.time() - start_time
    print(f"Total evaluation time: {total_time:.2f} seconds")
    
    # Process results
    for game_name, status, results, summary in results_list:
        finalResults[game_name] = status
        if results is not None:
            all_results.append(results)
        if summary is not None:
            all_summaries.append(summary)
if all_results:
    # Save summary to separate file
    summary_path = os.path.join(output_folder, f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    with open(summary_path, 'w') as f:
        f.write('\n'.join(all_summaries))

for game_name, status in finalResults.items():
    print(f"{game_name}: {status}")

distribution = Counter(finalResults.values())
print("\nDistribution of results:")
for status, count in distribution.items():
    print(f"{status}: {count}")