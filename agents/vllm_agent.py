import re
import sys
import os
import logging
from typing import List, Dict, Any, Optional

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from agents.agent import Agent

try:
    from vllm import LLM, SamplingParams
except ImportError:
    logging.warning("WARNING: vLLM not installed. Install with 'pip install vllm'")
    LLM, SamplingParams = None, None

class VLLMAgent(Agent):
    """Agent using vLLM for LLM-based game play"""
    
    def __init__(self, llm_engine, sampling_params, tokenizer, name: str = "VLLMAgent"):
        super().__init__(name)
        self.llm_engine = llm_engine
        self.sampling_params = sampling_params
        self.tokenizer = tokenizer
        
    def get_move(self, game, player_value) -> Any:
        """
        Get a move from the LLM
        
        Args:
            game: Game object
            player_value: Agent's player value in the game
            
        Returns:
            Move object (format depends on game implementation)
        """
        prompt_text, legal_moves = self._prepare_prompt(game)
        if not prompt_text or not legal_moves:
            return None
            
        # Generate response from LLM
        outputs = self.llm_engine.generate([prompt_text], self.sampling_params, use_tqdm=False)
        raw_output = outputs[0].outputs[0].text
        
        # Parse the move from the output - now using game's parse method
        move = game.parse_move_from_output(raw_output, legal_moves)
        return move
    
    def supports_batch(self) -> bool:
        """vLLM agent supports batch processing"""
        return True
    
    def get_batch_moves(self, game_contexts: List[Dict]) -> List[Any]:
        """
        Get moves for multiple games in batch
        
        Args:
            game_contexts: List of dictionaries containing:
                           - 'game': Game object
                           - 'player_value': Agent's player value
                           
        Returns:
            List of moves, one for each game context
        """
        prompts = []
        legal_moves_list = []
        valid_indices = []
        game_list = []
        
        # Prepare prompts for all valid games
        for i, context in enumerate(game_contexts):
            game = context.get('game')
            prompt_text, legal_moves = self._prepare_prompt(game)
            if prompt_text and legal_moves:
                prompts.append(prompt_text)
                legal_moves_list.append(legal_moves)
                valid_indices.append(i)
                game_list.append(game)  # Store the game object for parsing
        
        # If no valid prompts, return None for all contexts
        if not prompts:
            return [None] * len(game_contexts)
            
        # Generate responses in batch
        outputs = self.llm_engine.generate(prompts, self.sampling_params, use_tqdm=False)
        
        # Process the outputs and create result list
        results = [None] * len(game_contexts)
        for output_idx, output in enumerate(outputs):
            raw_output = output.outputs[0].text
            context_idx = valid_indices[output_idx]
            legal_moves = legal_moves_list[output_idx]
            game = game_list[output_idx]  # Get the corresponding game
            move = game.parse_move_from_output(raw_output, legal_moves)  # Use game's parse method
            results[context_idx] = move
            
        return results
    
    def _prepare_prompt(self, game):
        """
        Prepare the prompt for the LLM
        
        Args:
            game: Game object
            
        Returns:
            Tuple of (prompt_text, legal_moves)
        """
        try:
            messages = game.get_messages_for_llm()
            legal_moves = game.get_legal_moves()
            
            if not legal_moves:
                return None, []
                
            # Use tokenizer to format chat messages
            prompt_text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            return prompt_text, legal_moves
        except Exception as e:
            logging.error(f"Error preparing prompt: {e}")
            return None, []