import os
import sys
import logging
from typing import List, Dict, Any, Optional
from games.game import Game

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from agents.agent import Agent

# OpenAI API imports
try:
    from openai import OpenAI
    import openai
    import httpx
except ImportError:
    logging.warning("WARNING: OpenAI Python package not installed. Install with 'pip install openai'")
    OpenAI = None

class APIAgent(Agent):
    """Agent using external API services like OpenAI"""
    
    def __init__(self, api_client=None, model="gpt-4-0125-preview", api_base_url=None, api_key=None, name="APIAgent"):
        """
        Initialize an APIAgent
        
        Args:
            api_client: Pre-configured OpenAI client (optional)
            model: Model name to use (default: gpt-4-0125-preview)
            api_base_url: Base URL for API (if not using default OpenAI)
            api_key: API key (if not using a pre-configured client)
            name: Agent name
        """
        super().__init__(name)
        self.model = model
        
        if api_client is not None:
            self.client = api_client
        elif api_base_url is not None and api_key is not None:
            self.client = OpenAI(
                base_url=api_base_url,
                api_key=api_key,
                http_client=httpx.Client(
                    base_url=api_base_url,
                    follow_redirects=True,
                ),
                timeout=httpx.Timeout(600, read=600, write=600, connect=600),
            )
        else:
            if OpenAI is None:
                raise ImportError("OpenAI Python package is required but not installed. Install with 'pip install openai'")
            self.client = OpenAI()  # Use default client with API key from environment
            
        self.system_message = "You are a helpful assistant who is an expert at playing games."
    
    def get_move(self, game:Game) -> Any:
        """
        Get a move from the API service
        
        Args:
            game: Game object
            
        Returns:
            Move object (format depends on game implementation)
        """
        # Get prompt and legal moves
        messages = game.get_chat_history_for_llm(self)
        legal_moves = game.get_legal_moves()
        
        if not legal_moves:
            return None, "No legal moves available"
            
        # Generate response from API
        try:
            response = self.get_prompt_stream(messages)
            
            return response
        except Exception as e:
            logging.error(f"Error getting move from API: {e}")
            return "Error getting move from API"
    
    def get_prompt_stream(self, messages: List[Dict[str, str]]) -> str:
        """
        Use API to get streaming response
        
        Args:
            prompt: The prompt to send to the API
            
        Returns:
            The full response text
        """
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=True,
                timeout=600
            )
            
            collected_chunks = []
            for chunk in completion:
                if chunk.choices[0].delta.content is not None:
                    collected_chunks.append(chunk.choices[0].delta.content)
            
            full_response = ''.join(collected_chunks)
            return full_response
            
        except openai.InternalServerError as e:
            if 'timeout' in str(e).lower():
                logging.error("API timeout occurred")
                return None
            raise