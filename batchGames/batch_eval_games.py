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
from datetime import datetime

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
        game_class_info: tuple of (module_name, class_name, folder, output_folder, success_folder, TIME_LIMIT)
                         Backward-compat: (game_class, folder, output_folder, success_folder, TIME_LIMIT)
    
    Returns:
        tuple: (game_name, status, results_data, summary_data)
    """
    # Unpack and import in the worker to be spawn-safe
    if isinstance(game_class_info[0], str):
        module_name, class_name, folder, output_folder, success_folder, TIME_LIMIT = game_class_info
        sys.path.insert(0, folder)
        try:
            module = __import__(module_name, fromlist=[class_name])
            game_class = getattr(module, class_name)
        except Exception as e:
            print(f"Worker failed to import {module_name}.{class_name}: {e}")
            return (class_name, BasicResultStatus.FAILED_TO_IMPORT.value, None, None)
    else:
        game_class, folder, output_folder, success_folder, TIME_LIMIT = game_class_info
        class_name = game_class.__name__

    game_name = class_name
    
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
        os.makedirs(output_folder, exist_ok=True)
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
            try:
                with open(os.path.join(folder, f"{game_name}.py"), 'r') as f:
                    game_code = f.read()
                os.makedirs(success_folder, exist_ok=True)
                with open(os.path.join(success_folder, f"{game_name}.py"), 'w') as f:
                    f.write(game_code)
            except Exception:
                pass
        
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


def evaluate_games_in_folder(folder: str, output_folder: str, success_folder: str, TIME_LIMIT: int = 300, num_processes: int | None = None):
    """Programmatically evaluate all games in a folder using multiprocessing.

    Returns a dict of {game_name: status} and writes progress/summary to output_folder.
    """
    import time
    os.makedirs(folder, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(success_folder, exist_ok=True)

    # Discover game modules (assume class name == module name)
    sys.path.insert(0, folder)
    module_names = []
    finalResults = {}

    for _, module_name, _ in pkgutil.iter_modules([folder]):
        module_names.append(module_name)

    # Prepare arguments for multiprocessing: pass names, import inside worker
    game_args = [(name, name, folder, output_folder, success_folder, TIME_LIMIT) for name in module_names]

    # Determine number of processes
    if num_processes is None:
        num_processes = max(1, min(mp.cpu_count(), len(game_args)))

    print(f"Using {num_processes} processes for parallel evaluation")

    # Use multiprocessing to evaluate games in parallel
    results_list = []
    start_time = time.time()

    if len(game_args) > 0:
        # Set start method once
        if mp.get_start_method(allow_none=True) is None:
            mp.set_start_method('spawn')
        try:
            with Pool(processes=num_processes) as pool:
                try:
                    results_iter = pool.imap_unordered(evaluate_single_game, game_args)
                    completed = 0
                    for result in results_iter:
                        results_list.append(result)
                        completed += 1
                        elapsed = time.time() - start_time
                        game_name = result[0]
                        status = result[1]
                        text=f"[{completed}/{len(game_args)}] Completed {game_name}: {status} (elapsed: {elapsed:.1f}s)"
                        with open(os.path.join(output_folder, "progress.txt"), 'a') as f:
                            f.write(text + '\n')
                        print(text)
                        distribution = Counter([r[1] for r in results_list])
                        print("\nDistribution of results:")
                        for s, count in distribution.items():
                            print(f"{s}: {count}")
                except KeyboardInterrupt:
                    print("Interrupted by user. Terminating pool...")
                    pool.terminate()
                except Exception as e:
                    print(f"Error during multiprocessing: {e}")
                    pool.terminate()
                finally:
                    pool.join()
        except RuntimeError as e:
            # Start method already set in this interpreter
            print(f"RuntimeError initializing multiprocessing (non-fatal): {e}")
            results_list = []
    else:
        print("No game modules found to evaluate.")

    total_time = time.time() - start_time
    print(f"Total evaluation time: {total_time:.2f} seconds")

    all_results = []
    all_summaries = []
    for game_name, status, results, summary in results_list:
        finalResults[game_name] = status
        if results is not None:
            all_results.append(results)
        if summary is not None:
            all_summaries.append(summary)

    if all_summaries:
        summary_path = os.path.join(output_folder, f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        with open(summary_path, 'w') as f:
            f.write('\n'.join(all_summaries))

    for game_name, status in finalResults.items():
        print(f"{game_name}: {status}")

    distribution = Counter(finalResults.values())
    print("\nDistribution of results:")
    for status, count in distribution.items():
        print(f"{status}: {count}")

    return finalResults


if __name__ == "__main__":
    # Default paths under debug/
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    base = os.path.join(project_root, 'debug', f'batch_eval_{ts}')
    games_folder = os.path.join(base, 'games')
    out_folder = os.path.join(base, 'eval')
    succ_folder = os.path.join(base, 'success')
    os.makedirs(games_folder, exist_ok=True)
    os.makedirs(out_folder, exist_ok=True)
    os.makedirs(succ_folder, exist_ok=True)

    # Optional CLI args: folder out_folder success_folder time_limit processes
    args = sys.argv[1:]
    if len(args) >= 1: games_folder = args[0]
    if len(args) >= 2: out_folder = args[1]
    if len(args) >= 3: succ_folder = args[2]
    time_limit = int(args[3]) if len(args) >= 4 else 300
    processes = int(args[4]) if len(args) >= 5 else None

    evaluate_games_in_folder(games_folder, out_folder, succ_folder, time_limit, processes)