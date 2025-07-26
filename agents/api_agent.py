import os
import sys
import logging
import concurrent.futures
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
    
    def __init__(self, api_client=None, model="gpt-4-0125-preview", api_base_url=None, api_key=None, name="Default"):
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
        
        if name=="Default":
            self.name = f"API Agent ({model})"
        
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
    
    def get_batch_moves(self, game_contexts: List[Dict]) -> List[str]:
        """
        Get moves for multiple games in batch using OpenAI batch completion
        
        Args:
            game_contexts: List of dictionaries with:
                           - 'game': Game object
                           
        Returns:
            List[str]: List of raw LLM outputs
        """
        if not game_contexts:
            return []
            
        # Prepare all prompts for batch processing
        all_messages = []
        for context in game_contexts:
            game = context.get('game')
            if game:
                messages = game.get_chat_history_for_llm(self)
                all_messages.append(messages)
            else:
                all_messages.append([])
        
        # Use batch completion if available, otherwise fall back to concurrent processing
        if hasattr(self.client.chat.completions, 'create_batch'):
            return self._get_batch_moves_with_batch_api(all_messages)
        else:
            return self._get_batch_moves_concurrent(all_messages)
    
    def _get_batch_moves_with_batch_api(self, all_messages: List[List[Dict[str, str]]]) -> List[str]:
        """
        Use OpenAI's native batch API for efficient batch processing
        
        Args:
            all_messages: List of message arrays for each game
            
        Returns:
            List[str]: List of raw LLM outputs
        """
        try:
            # Prepare batch requests
            batch_requests = []
            for i, messages in enumerate(all_messages):
                if messages:
                    batch_requests.append({
                        "custom_id": f"request-{i}",
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": {
                            "model": self.model,
                            "messages": messages,
                            "timeout": 600
                        }
                    })
            
            # Create batch
            batch_response = self.client.chat.completions.create_batch(
                requests=batch_requests
            )
            
            # Extract responses in order
            responses = [""] * len(all_messages)
            for response in batch_response.responses:
                idx = int(response.custom_id.split('-')[1])
                if response.body and response.body.choices:
                    responses[idx] = response.body.choices[0].message.content
                else:
                    responses[idx] = "Error in batch response"
            
            return responses
            
        except Exception as e:
            logging.error(f"Error in batch API: {e}")
            # Fall back to concurrent processing
            return self._get_batch_moves_concurrent(all_messages)
    
    def _get_batch_moves_concurrent(self, all_messages: List[List[Dict[str, str]]]) -> List[str]:
        """
        Use concurrent requests for batch processing
        
        Args:
            all_messages: List of message arrays for each game
            
        Returns:
            List[str]: List of raw LLM outputs
        """
        import concurrent.futures
        import threading
        
        results = [""] * len(all_messages)
        
        def get_single_response(index, messages):
            try:
                if not messages:
                    results[index] = "No messages provided"
                    return
                    
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    timeout=600
                )
                results[index] = response.choices[0].message.content
            except Exception as e:
                logging.error(f"Error getting move for index {index}: {e}")
                results[index] = f"Error getting move: {str(e)}"
        
        # Use ThreadPoolExecutor for concurrent requests
        max_workers = min(len(all_messages), 30)  # Limit concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for i, messages in enumerate(all_messages):
                future = executor.submit(get_single_response, i, messages)
                futures.append(future)
            
            # Wait for all futures to complete
            concurrent.futures.wait(futures)
        
        return results

    def get_prompt_stream(self, messages: List[Dict[str, str]]) -> str:
        """
        Use API to get streaming response
        
        Args:
            messages: The messages to send to the API
            
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
    
    def get_prompt_non_stream(self, messages: List[Dict[str, str]]) -> str:
        """
        Use API to get non-streaming response (useful for batch processing)
        
        Args:
            messages: The messages to send to the API
            
        Returns:
            The response text
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                timeout=600
            )
            
            return response.choices[0].message.content
            
        except openai.InternalServerError as e:
            if 'timeout' in str(e).lower():
                logging.error("API timeout occurred")
                return None
            raise