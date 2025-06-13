import re
import sys
import os
import logging
from typing import List, Dict, Any, Optional, Tuple

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
    
    def get_move(self, game) -> str:
        """
        Get a move from the LLM
        
        Args:
            game: Game object
            
        Returns:
            str: Raw LLM output string (unparsed)
        """
        prompt_text, legal_moves = self._prepare_prompt(game)
        if not prompt_text:
            return "Error preparing prompt"
        if not legal_moves:
            return "No legal moves available"
            
        # Generate response from LLM
        outputs = self.llm_engine.generate([prompt_text], self.sampling_params, use_tqdm=False)
        raw_output = outputs[0].outputs[0].text
        
        # Return raw output without any parsing
        return raw_output
    
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