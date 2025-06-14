# ChessBattleAssessment

A comprehensive evaluation framework for AI agents playing Chess-like Board Games (currently TicTacToe and Connect 4), supporting multiple agent types including API-based models, local language models, and traditional algorithms.

## Features

- **Multiple Agent Types**: Support for API agents, vLLM agents, Random agents, and Minimax agents
- **Multiple Game Types**: TicTacToe and Connect 4 support
- **Human vs Agent Mode**: Interactive gameplay against AI agents
- **Flexible Evaluation**: Evaluate any agent against any other agent or multiple opponents
- **Comprehensive Logging**: Detailed game logs and performance metrics
- **Configurable Settings**: Customizable game parameters and evaluation settings

## Project Structure

```
ChessBattleAssessment/
├── agents/                          # Agent implementations
│   ├── api_agent.py                 # API-based agents (OpenAI, etc.)
│   ├── vllm_agent.py                # Local vLLM model agents
│   ├── random_agent.py              # Random move agent
│   ├── minimax_agent_connect4.py    # Minimax algorithm agent for connect 4 game
│   ├── minimax_agent_tictactoe.py   # Minimax algorithm agent for tictactoe game
│   └── minimax_agent.py             # Minimax algorithm agent, route to above subagents
├── games/                           # Game implementations
│   ├── tictactoe.py                 # TicTacToe game logic
│   └── connect4.py                  # Connect 4 game logic
├── evaluation/                      # Evaluation framework
│   └── evaluator.py                 # Agent evaluation system
├── utils/                           # Utility modules
│   └── model_utils.py               # Model loading and configuration
├── config.py                        # Configuration settings
├── evaluate_two_agents.py           # Two-agent battle evaluation
├── human_vs_agent.py                # Human vs agent interactive mode
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
- Uses minimax algorithm with alpha-beta pruning
- Perfect play for TicTacToe
- Deterministic behavior
- Strong baseline opponent

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
pip install torch transformers
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
    --num_games 100
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
    --num_games 50
```

2. **Two Different API Models**
```bash
python evaluate_two_agents.py \
    --agent1 api --agent1_model gpt-4 --agent1_api_base_url URL1 --agent1_api_key KEY1 \
    --agent2 api --agent2_model claude-3 --agent2_api_base_url URL2 --agent2_api_key KEY2 \
    --num_games 50
```

3. **Local Model vs Random**
```bash
python evaluate_two_agents.py \
    --agent1 vllm --agent1_model_path /path/to/model \
    --agent2 random \
    --num_games 100
```

## Human vs Agent Mode

To play against an AI agent:

```bash
python human_vs_agent.py \
    --agent api \
    --agent_model gpt-4 \
    --agent_api_base_url https://api.openai.com/v1 \
    --agent_api_key your_key \
    --game tictactoe
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

