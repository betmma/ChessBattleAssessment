import re
import sys
import os
import logging
from typing import List, Dict, Any, Optional, Tuple
from games.game import Game

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
    
    def __init__(self, llm_engine, sampling_params, tokenizer, name: str = "VLLMAgent", enable_thinking: bool = True):
        super().__init__(name)
        self.llm_engine = llm_engine
        self.sampling_params = sampling_params
        self.tokenizer = tokenizer
        self.enable_thinking = enable_thinking
    
    def get_move(self, game:Game) -> str:
        """
        Get a move from the LLM
        
        Args:
            game: Game object
            
        Returns:
            str: Raw LLM output string (unparsed)
        """
        prompt_text = self._prepare_prompt(game)
        if not prompt_text:
            return "Error preparing prompt"
            
        # Generate response from LLM
        outputs = self.llm_engine.generate([prompt_text], self.sampling_params, use_tqdm=False)
        raw_output = outputs[0].outputs[0].text
        
        # Return raw output without any parsing
        return raw_output
    
    def get_batch_moves(self, game_contexts: List[Dict]) -> List[str]:
        """
        Get moves for a batch of game contexts
        
        Args:
            game_contexts: List of game context dictionaries
            
        Returns:
            List[str]: Raw LLM outputs for each game context
        """
        prompts = []
        
        for game in game_contexts:
            prompt_text = self._prepare_prompt(game)
            if not prompt_text:
                prompts.append("Error preparing prompt")
            else:
                prompts.append(prompt_text)
        
        # Generate responses from LLM
        outputs = self.llm_engine.generate(prompts, self.sampling_params, use_tqdm=False)
        
        # Return raw outputs without any parsing
        return [output.outputs[0].text for output in outputs]
        
    
    def _prepare_prompt(self, game:Game):
        """
        Prepare the prompt for the LLM
        
        Args:
            game: Game object
            
        Returns:
            Tuple of (prompt_text, legal_moves)
        """
        try:
            messages = game.get_chat_history_for_llm(self)
                
            # Use tokenizer to format chat messages
            prompt_text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=self.enable_thinking
            )
            
            return prompt_text
        except Exception as e:
            logging.error(f"Error preparing prompt: {e}")
            return None