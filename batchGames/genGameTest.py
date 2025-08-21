import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,3"
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import time
import random
import shutil
import json
from datetime import datetime
from pathlib import Path
import re
import sys

# Ensure we can import the evaluator function
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from batchGames.batch_eval_games import evaluate_games_in_folder, BasicResultStatus

# --- Config ---
MODEL_PATH = '/remote-home1/share/models/Qwen3-32B'
MAX_MODEL_LEN = 16384
TEMPERATURE = 0.7
TOP_P = 0.95
TOP_K = 40
MAX_TOKENS = 16384
POP_SIZE = 200
ELITE_K = 20
GENERATIONS = 5
TIME_LIMIT_PER_GAME = 240
PROCESSES = None  # auto
WORDS_PATH = '/remote-home1/yrmou/1krdwords.txt'
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_TS = datetime.now().strftime('%Y%m%d_%H%M%S')
RUN_BASE = PROJECT_ROOT / 'debug' / f'evo_games_{RUN_TS}'
(GENERATION_DIR := RUN_BASE / 'generations').mkdir(parents=True, exist_ok=True)
(RUN_BASE / 'logs').mkdir(parents=True, exist_ok=True)

# Load board game base class
BOARD_GAME_PATH = PROJECT_ROOT / 'games' / 'board_game.py'
with open(BOARD_GAME_PATH, 'r') as f:
    BOARD_GAME_CODE = f.read()

# Initialize tokenizer and LLM
print(f"Loading model from {MODEL_PATH}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
llm = LLM(model=MODEL_PATH, max_model_len=MAX_MODEL_LEN, tensor_parallel_size=2)
sampling_params = SamplingParams(temperature=TEMPERATURE, top_p=TOP_P, top_k=TOP_K, max_tokens=MAX_TOKENS)

SYSTEM_PROMPT = "You are a game designer and expert programmer."
BASE_USER_PROMPT = (
    "Write a game in Python. The game should be a subclass of the BoardGame class. "
    "BoardGame class is as below:\n```\n" + BOARD_GAME_CODE + "\n```\n"
    "The game must be BRAND NEW and UNIQUE. It must not be a simple placement game; prefer abstract, perfect-information rules like Nim. "
    "Design a compact state with around 10 legal moves on average and an average game length near 20 turns. "
    "All information must be on the board. Your output must ONLY include the class of your game, beginning with:\n```") + \
    "\nimport numpy as np\nfrom typing import List, Any, Optional, Tuple\nfrom games.board_game import BoardGame\n" + \
    "```\nDo NOT repeat the BoardGame class. Incorporate the brainstorming word: "

# Mutation helpers
MUTATION_INSTRUCTIONS = [
    # --- Board Structure & Initialization Mutations ---
    "Change the number of dimensions of the game state (e.g., 1D list -> 2D grid).",
    "Increase or decrease the size of one of the board's dimensions by a small amount.",
    "Change the initial value of game pieces (e.g., from a constant '5' to a random value in a range).",
    "Modify the initialization pattern (e.g., from 'all piles have N items' to 'piles have 1, 3, 5... items').",
    "Introduce or remove non-uniformity in the initial state (e.g., create a ragged array or a board with initial 'holes').",

    # --- Move Rule Mutations: Target Selection ---
    "Change the number of distinct locations a player can select in one turn (e.g., from 1 pile to 2).",
    "Modify the type of selectable target (e.g., from 'a single piece/cell' to 'an entire row/column').",
    "Add a conditional restriction to what can be selected (e.g., 'cannot select a piece with value 0' or 'must select the largest pile').",
    "Remove a restriction on what can be selected.",

    # --- Move Rule Mutations: Action/Effect ---
    "Change the primary move operation (e.g., from 'decrement value' to 'set value to zero', 'halve value', or 'flip boolean state').",
    "Alter the magnitude of the move's effect (e.g., from 'remove any amount > 0' to 'remove exactly 1').",
    "Add or remove a 'cascade' or 'area of effect' rule (like in Chomp, where selecting one piece affects others).",
    "If a cascade effect exists, change its shape or direction (e.g., from 'all pieces to the bottom-right' to 'all pieces in the same row').",
    "If a cascade effect exists, change its logic (e.g., from 'remove affected pieces' to 'decrement affected pieces').",

    # --- Termination Condition Mutations ---
    "Change the termination condition from 'state is empty' to 'sum of state values is below a threshold X'.",
    "Change the termination condition from 'no valid moves remain' to 'a specific piece/cell is taken'.",
    "Add a turn limit to the game.",
    "Modify the termination condition to trigger when only one non-zero element/region remains.",

    # --- Win/Loss Condition Mutations ---
    "Flip the win condition from Normal Play to Misère Play (last player to move wins -> loses).",
    "Flip the win condition from Misère Play to Normal Play (last player to move loses -> wins).",
    "Designate a specific piece or cell as a 'poison pill'; taking it results in an immediate loss.",
    "If a 'poison pill' exists, change its location or remove it.",
    "Change the win condition from being based on the last move to being based on the final board state score.",
    
    # --- Advanced/Complex Mutations ---
    "Add a new, secondary move type with different rules, available under certain conditions.",
    "Split one game piece/pile into two smaller ones as a valid move.",
    "Merge two adjacent game pieces/piles into one as a valid move.",
    "Introduce a 'pass turn' option, possibly at a cost or only if no other move is available.",
    "Couple the state of two different pieces (e.g., 'any action on piece A also applies to piece B').",
]

BANNED_PHRASES = [
    "Connect4", "TicTacToe", "placing pieces", "checkerboard", "grid placement", "Go", "Chess"
]

def apply_chat(system_prompt, user_prompt, thinking=True):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=thinking,
    )


def generate_candidates(seed_words, out_dir):
    prompts = []
    for w in seed_words:
        prompts.append(BASE_USER_PROMPT + w)
    texts = [apply_chat(SYSTEM_PROMPT, p, thinking=True) for p in prompts]
    outputs = llm.generate(texts, sampling_params)
    results = []
    for p, out in zip(prompts, outputs):
        gen = out.outputs[0].text
        results.append((p, gen))
    # Save raw
    (out_dir / 'raw').mkdir(parents=True, exist_ok=True)
    with open(out_dir / 'raw' / 'raw_generations.txt', 'w') as f:
        for p, gen in results:
            f.write(f"PROMPT:\n{p}\n\nGEN:\n{gen}\n\n===\n")
    return results


def extract_class_name(code_text):
    m = re.search(r"class\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(\s*BoardGame", code_text)
    return m.group(1) if m else None


def sanitize_and_save_candidates(results, games_dir):
    games_dir.mkdir(parents=True, exist_ok=True)
    kept = []
    seen_names = set()
    for _, gen in results:
        gen=gen.split('</think>')[-1].strip()
        if any(bad.lower() in gen.lower() for bad in BANNED_PHRASES):
            continue
        # Extract code block containing import header
        m = re.search(r"```[\s\S]*?import numpy as np[\s\S]*?from games\.board_game import BoardGame[\s\S]*?```", gen)
        code = None
        if m:
            block = m.group(0)
            code = block.strip('`')
            if code.startswith('python'):
                code = code[6:]
        else:
            # Fallback: try to find from header line to end
            hdr = "import numpy as np\nfrom typing import List, Any, Optional, Tuple\nfrom games.board_game import BoardGame"
            pos = gen.find(hdr)
            if pos != -1:
                code = gen[pos:]
        if not code:
            continue
        class_name = extract_class_name(code)
        if not class_name or class_name in seen_names:
            continue
        seen_names.add(class_name)
        file_path = games_dir / f"{class_name}.py"
        with open(file_path, 'w') as f:
            f.write(code)
        kept.append((class_name, file_path))
    return kept


def evaluate_population(games_dir, eval_dir, success_dir):
    try:
        return evaluate_games_in_folder(str(games_dir), str(eval_dir), str(success_dir), TIME_LIMIT_PER_GAME, PROCESSES)
    except Exception as e:
        print(f"Error evaluating population: {e}")
        return {}


def compute_fitness(status_map, eval_dir):
    # Lower is better penalty; SUCCESS best
    order = {
        BasicResultStatus.SUCCESS.value: 0,
        BasicResultStatus.QUICK_END_IN_4.value: 3,
        BasicResultStatus.ALL_DRAW.value: 4,
        BasicResultStatus.ANY_FORFEIT.value: 5,
        BasicResultStatus.ALL_FORFEIT.value: 8,
        BasicResultStatus.ALL_SAME.value: 9,
        BasicResultStatus.ALL_FIRST_PLAYER_WIN.value: 6,
        BasicResultStatus.ALL_SECOND_PLAYER_WIN.value: 6,
        BasicResultStatus.NO_FIRST_PLAYER_WIN.value: 7,
        BasicResultStatus.NO_SECOND_PLAYER_WIN.value: 7,
        BasicResultStatus.TIME_LIMIT_EXCEEDED.value: 10,
        BasicResultStatus.FAILED_TO_IMPORT.value: 15,
        BasicResultStatus.FAILED_TO_EVALUATE.value: 15,
    }
    fitness = {name: -order.get(status, 20) for name, status in status_map.items()}
    # Optionally use summary metrics for tie-breakers
    return fitness


def mutate_prompt(prompt):
    tweak = random.choice(MUTATION_INSTRUCTIONS)
    return prompt + "\nRefine with this constraint: " + tweak


def select_and_breed(prompts, status_map, fitness, k):
    # Map class_name back to prompt by parsing class name from gens is hard; here we random sample top ones
    # Use fitness ranking mapped by name substring if present; fallback to random
    ranked = sorted(fitness.items(), key=lambda x: x[1], reverse=True)
    top_names = set([n for n, _ in ranked[:k]])
    # Keep prompts that mention class names; otherwise keep best k prompts by position
    survivors = prompts[:k]
    # Mutate to fill back to POP_SIZE
    children = []
    while len(survivors) + len(children) < POP_SIZE:
        base = random.choice(survivors)
        children.append(mutate_prompt(base))
    return survivors + children


def main():
    random.seed(42)
    # Seed words
    if os.path.exists(WORDS_PATH):
        with open(WORDS_PATH, 'r') as f:
            words = f.read().splitlines()
    else:
        words = [
            'entropy', 'lattice', 'cascade', 'flux', 'cipher', 'quorum', 'oracle', 'quiver', 'tessellate', 'ember',
            'delta', 'sigma', 'zenith', 'aperture', 'keystone', 'axiom', 'braid', 'glyph', 'matrix', 'quanta'
        ]
    population_words = random.sample(words, POP_SIZE)
    prompts = [BASE_USER_PROMPT + w for w in population_words]

    for gen in range(GENERATIONS):
        gen_dir = GENERATION_DIR / f'gen_{gen:02d}'
        games_dir = gen_dir / 'games'
        eval_dir = gen_dir / 'eval'
        succ_dir = gen_dir / 'success'
        (gen_dir / 'raw').mkdir(parents=True, exist_ok=True)
        print(f"=== Generation {gen} ===")
        # Generate
        results = generate_candidates([w for w in population_words], gen_dir)
        # Save prompts
        with open(gen_dir / 'prompts.json', 'w') as f:
            json.dump([p for p, _ in results], f, indent=2)
        # Materialize games
        kept = sanitize_and_save_candidates(results, games_dir)
        print(f"Saved {len(kept)} candidate games to {games_dir}")
        # Evaluate safely
        try:
            status_map = evaluate_population(games_dir, eval_dir, succ_dir)
        except KeyboardInterrupt:
            print("Interrupted. Stopping evolutionary run.")
            break
        except Exception as e:
            print(f"Evaluation failed for generation {gen}: {e}")
            with open(gen_dir / 'error.txt', 'w') as f:
                f.write(str(e))
            # Do not create a new generation on failure
            break
        with open(gen_dir / 'status.json', 'w') as f:
            json.dump(status_map, f, indent=2)
        # Fitness
        fitness = compute_fitness(status_map, eval_dir)
        with open(gen_dir / 'fitness.json', 'w') as f:
            json.dump(fitness, f, indent=2)
        # Prepare next gen prompts
        prompts = select_and_breed(prompts, status_map, fitness, ELITE_K)
        # Update seed words for logging/debug
        population_words = [f"mut_{i}" for i in range(len(prompts))]

    print(f"Run completed. Artifacts in {RUN_BASE}")


if __name__ == '__main__':
    main()

