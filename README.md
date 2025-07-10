# ChessBattleAssessment

A comprehensive evaluation framework for AI agents playing Chess-like Board Games (currently TicTacToe, Connect 4, and Nim), supporting multiple agent types including API-based models, local language models, and traditional algorithms.

## Features

- **Multiple Agent Types**: Support for API agents, vLLM agents, Random agents, and Minimax agents.
- **Multiple Game Types**: TicTacToe, Connect 4 and Nim support.
- **Human vs Agent Mode**: Interactive gameplay against AI agents.
- **Flexible Evaluation**: Evaluate any agent against any other agent or multiple opponents.
- **Comprehensive Logging**: Detailed game logs and performance metrics.
- **Configurable Settings**: Customizable game parameters and evaluation settings.
- **GRPO Dataset Generation**: Generate datasets for GRPO (Generative Representational Policy Optimization) from game logs.

## Project Structure

```
ChessBattleAssessment/
├── agents/                          # Agent implementations
│   ├── api_agent.py                 # API-based agents (OpenAI, etc.)
│   ├── vllm_agent.py                # Local vLLM model agents
│   ├── random_agent.py              # Random move agent
│   ├── minimax_agent.py             # Wrap universal_minimax_agent to add random move
│   └── universal_minimax_agent.py   # Universal Minimax agent for all games
├── games/                           # Game implementations
│   ├── tictactoe.py                 # TicTacToe game logic
│   ├── connect4.py                  # Connect 4 game logic
│   └── nim.py                       # Nim game logic
├── evaluation/                      # Evaluation framework
│   └── evaluator.py                 # Agent evaluation system
├── grpo/                            # GRPO dataset generation scripts and results
├── utils/                           # Utility modules
│   └── model_utils.py               # Model loading and configuration
├── config.py                        # Configuration settings
├── evaluate_two_agents.py           # Two-agent battle evaluation
├── play_against_agent.py            # Human vs agent interactive mode
├── generate_grpo_dataset.py         # Script to generate GRPO dataset
└── README.md                        # This file
```

## Game Types

### 1. TicTacToe
- Classic 3x3 grid game
- First to get three in a row wins
- Perfect play solvable

### 2. Connect 4
- 6x7 grid with gravity-based moves
- First to get four in a row (horizontal, vertical, or diagonal) wins
- More complex strategy than TicTacToe

### 3. Nim
- Mathematical game of strategy
- Players take turns removing objects from distinct heaps
- The player to take the last object wins or loses depending on the version

## Agent Types

### 1. API Agent
- Connects to external API services (OpenAI, Claude, etc.)
- Configurable model selection
- Supports custom API endpoints
- Timeout and retry handling

### 2. vLLM Agent
- Uses local language models via vLLM
- GPU acceleration support
- Configurable sampling parameters
- Memory efficient inference

### 3. Random Agent
- Makes random valid moves
- Useful as baseline for evaluation
- No configuration required

### 4. Minimax Agent
- Wraps Universal Minimax Agent to add random move

### 5. Universal Minimax Agent
- A minimax agent that can play any of the supported games.
- Used for dataset generation.

## Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd ChessBattleAssessment
```

2. **Install dependencies**
```bash
# For API agents
pip install openai httpx

# For vLLM agents
pip install vllm

# Other dependencies
pip install torch transformers tqdm
```

3. **Set up CUDA (optional, for GPU acceleration)**
```bash
export CUDA_VISIBLE_DEVICES="0"  # Adjust GPU ID as needed
```

## Usage

### 1. Two-Agent Battle

Evaluate any two agents against each other:

```bash
python evaluate_two_agents.py \
    --agent1 api \
    --agent1_model gpt-4 \
    --agent1_api_base_url https://api.openai.com/v1 \
    --agent1_api_key your_key \
    --agent2 minimax \
    --num_games 100 \
    --game tictactoe
```

**Agent Configuration:**

#### API Agent
```bash
--agent1 api \
--agent1_model gpt-4-0125-preview \
--agent1_api_base_url https://api.openai.com/v1 \
--agent1_api_key your_api_key
```

#### vLLM Agent
```bash
--agent1 vllm \
--agent1_model_path /path/to/your/model
```

#### Random Agent
```bash
--agent1 random
```

#### Minimax Agent
```bash
--agent1 minimax
```

### Example Battles

1. **API vs Minimax**
```bash
python evaluate_two_agents.py \
    --agent1 api --agent1_model gpt-4 --agent1_api_base_url URL --agent1_api_key KEY \
    --agent2 minimax \
    --num_games 50 \
    --game connect4
```

2. **Two Different API Models**
```bash
python evaluate_two_agents.py \
    --agent1 api --agent1_model gpt-4 --agent1_api_base_url URL1 --agent1_api_key KEY1 \
    --agent2 api --agent2_model claude-3 --agent2_api_base_url URL2 --agent2_api_key KEY2 \
    --num_games 50 \
    --game tictactoe
```

3. **Local Model vs Random**
```bash
python evaluate_two_agents.py \
    --agent1 vllm --agent1_model_path /path/to/model \
    --agent2 random \
    --num_games 100 \
    --game nim
```

### 2. Human vs Agent Mode

To play against an AI agent:

```bash
python play_against_agent.py \
    --agent api \
    --agent_model gpt-4 \
    --agent_api_base_url https://api.openai.com/v1 \
    --agent_api_key your_key \
    --game tictactoe
```

### 3. Generate GRPO Dataset

Generate a GRPO dataset from existing evaluation logs:
```bash
python generate_grpo_dataset.py \
    --input_file evaluation_results_vllm/CONSOLIDATED_summary_Minimax-random-0.0-depth-4_vs_Minimax-random-0.0-depth-4_20250709-195527.txt \
    --output_file grpo/dataset.jsonl \
    --max_depth 4
```

## Configuration

The `config.py` file contains default settings:

```python
class Config:
    NUM_EVAL_GAMES = 50        # Default number of games
    OUTPUT_DIR_BASE = "./results"  # Output directory
    MODEL_PATH = "/path/to/default/model"  # Default vLLM model path
    
    # vLLM settings
    TENSOR_PARALLEL_SIZE = 1
    TEMPERATURE = 0.7
    TOP_P = 0.95
    MAX_TOKENS = 1024
```

## Output

### Game Results
- Win/loss/draw statistics
- Performance metrics
- Detailed game logs
- Move history for each game

## Generate New Game
Use this prompt and send board_game.py to copilot:
based on this board game abstract class, design a BRAND NEW and UNIQUE board game. the file should be put inside games/boardGames. the number of possible states of the game should be around 1k-1m for minimax to work. the game must be perfect information.
a random word to help you brainstorm: sugar
tell me the design and only code after i approve
