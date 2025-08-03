import sys,os,pkgutil,json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import Config, setup_logging
import logging
from enum import Enum
from dataclasses import dataclass
import signal
from contextlib import contextmanager

from evaluate_board_game import BoardGameBalanceEvaluator
folder = '/root/myr/genGames/unique_games'
output_folder = '/root/myr/genGames/eval'
success_folder = '/root/myr/genGames/unique_games2'
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
    QUICK_END_IN_3 = "Under some agent pairs, all games end in 3 steps"
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
evaluator = BoardGameBalanceEvaluator(config=config, num_games=100, depth=3)
all_results = []
all_summaries = []
from datetime import datetime
TIME_LIMIT = 60  # seconds
for game_class in gameClasses:
    try:
        with timeout(TIME_LIMIT):  # 1 minute timeout
            results = evaluator.evaluate_all_metrics(game_class)
        summary = evaluator.format_summary(results)
        all_summaries.append(summary)
        output_path = evaluator.save_results(results, os.path.join(output_folder, f"{game_class.__name__}_evaluation.json"))
        all_results.append(results)
        
        metrics=results['agent_pair_metrics']
        def isForfeitAny(agent_pair_metric):
            result=agent_pair_metric['detailed_results']
            return result['forfeits_agent1']+result['forfeits_agent2'] >0
        def isForfeitAll(agent_pair_metric):
            result=agent_pair_metric['detailed_results']
            return result['forfeits_agent1']+result['forfeits_agent2'] == result['total_games']
        def isDrawAll(agent_pair_metric):
            result=agent_pair_metric['detailed_results']
            return result['draws'] == result['total_games']
        def isFirstPlayerWinAll(agent_pair_metric):
            balance= agent_pair_metric['balance']
            result=agent_pair_metric['detailed_results']
            return balance['first_player_wins'] == result['total_games']
        def isSecondPlayerWinAll(agent_pair_metric):
            balance= agent_pair_metric['balance']
            result=agent_pair_metric['detailed_results']
            return balance['second_player_wins'] == result['total_games']
        def isNoFirstPlayerWin(agent_pair_metric):
            balance= agent_pair_metric['balance']
            result=agent_pair_metric['detailed_results']
            return balance['first_player_wins'] == 0 and result['total_games'] > 0
        def isNoSecondPlayerWin(agent_pair_metric):
            balance= agent_pair_metric['balance']
            result=agent_pair_metric['detailed_results']
            return balance['second_player_wins'] == 0 and result['total_games'] > 0
        def isAllSame(agent_pair_metric):
            return agent_pair_metric["variation"]['unique_states'] == 1
        def length(agent_pair_metric):
            return agent_pair_metric["length"]['average_length']
        if all(isForfeitAll(metric) for metric in metrics.values()):
            finalResults[game_class.__name__] = BasicResultStatus.ALL_FORFEIT.value
        elif any(isForfeitAny(metric) for metric in metrics.values()):
            finalResults[game_class.__name__] = BasicResultStatus.ANY_FORFEIT.value
        elif all(isDrawAll(metric) for metric in metrics.values()):
            finalResults[game_class.__name__] = BasicResultStatus.ALL_DRAW.value
        elif all(isFirstPlayerWinAll(metric) for metric in metrics.values()):
            finalResults[game_class.__name__] = BasicResultStatus.ALL_FIRST_PLAYER_WIN.value
        elif all(isSecondPlayerWinAll(metric) for metric in metrics.values()):
            finalResults[game_class.__name__] = BasicResultStatus.ALL_SECOND_PLAYER_WIN.value
        elif all(isAllSame(metric) for metric in metrics.values()):
            finalResults[game_class.__name__] = BasicResultStatus.ALL_SAME.value
        elif all(isNoFirstPlayerWin(metric) for metric in metrics.values()):
            finalResults[game_class.__name__] = BasicResultStatus.NO_FIRST_PLAYER_WIN.value
        elif all(isNoSecondPlayerWin(metric) for metric in metrics.values()):
            finalResults[game_class.__name__] = BasicResultStatus.NO_SECOND_PLAYER_WIN.value
        elif any(length(metric) <= 3 for metric in metrics.values()):
            finalResults[game_class.__name__] = BasicResultStatus.QUICK_END_IN_3.value
        else:
            finalResults[game_class.__name__] = BasicResultStatus.SUCCESS.value
            # Save the game class source code to success folder
            with open(os.path.join(folder, f"{game_class.__name__}.py"), 'r') as f:
                game_code = f.read()
            with open(os.path.join(success_folder, f"{game_class.__name__}.py"), 'w') as f:
                f.write(game_code)
            
    except TimeoutError:
        finalResults[game_class.__name__] = BasicResultStatus.TIME_LIMIT_EXCEEDED.value
        print(f"Timeout exceeded for {game_class.__name__}: evaluation took longer than {TIME_LIMIT} seconds")
    except Exception as e:
        finalResults[game_class.__name__] = BasicResultStatus.FAILED_TO_EVALUATE.value
        import traceback
        traceback.print_exc()
        print(f"Error evaluating {game_class.__name__}: {e}")
if all_results:
    combined_results = {
        "evaluation_summary": {
            "timestamp": datetime.now().isoformat(),
            "num_games_per_metric": evaluator.num_games,
            "total_games_evaluated": len(all_results)
        },
        "results": all_results,
        "formatted_summaries": all_summaries
    }
    
    # combined_path = os.path.join(output_folder, f"combined_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    # with open(combined_path, 'w') as f:
    #     json.dump(combined_results, f, indent=2, default=str)
    
    # Save summary to separate file
    summary_path = os.path.join(output_folder, f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    with open(summary_path, 'w') as f:
        f.write('\n'.join(all_summaries))

for game_name, status in finalResults.items():
    print(f"{game_name}: {status}")

from collections import Counter
distribution = Counter(finalResults.values())
print("\nDistribution of results:")
for status, count in distribution.items():
    print(f"{status}: {count}")