import os
import logging
import torch
from transformers import AutoTokenizer

try:
    from vllm import LLM, SamplingParams
except ImportError:
    logging.warning("vLLM not installed. To use vLLM backend, please install with 'pip install vllm'")
    LLM, SamplingParams = None, None

class ModelUtils:
    """Utilities for initializing models and tokenizers"""
    
    @staticmethod
    def initialize_vllm_model(model_path, tensor_parallel_size=1):
        """
        Initialize vLLM model and tokenizer
        
        Args:
            model_path: Path to the model
            tensor_parallel_size: Number of GPUs for tensor parallelism
            
        Returns:
            Tuple of (llm_engine, sampling_params, tokenizer)
        """
        if LLM is None:
            raise ImportError("vLLM is required but not installed. Install with 'pip install vllm'")
            
        logging.info(f"Initializing vLLM engine for model: {model_path}")
        
        # Initialize tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        # Initialize vLLM engine
        llm_engine = LLM(
            model=model_path,
            tokenizer=model_path,
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=True,
        )
        
        # Create default sampling parameters
        sampling_params = SamplingParams(
            temperature=0.1,
            top_p=0.95,
            max_tokens=256,
            stop_token_ids=[tokenizer.eos_token_id] if tokenizer.eos_token_id else []
        )
        
        logging.info("vLLM Engine and Tokenizer initialized.")
        return llm_engine, sampling_params, tokenizer
    
    @staticmethod
    def update_sampling_params(sampling_params, temperature=None, top_p=None, max_tokens=None):
        """
        Update sampling parameters
        
        Args:
            sampling_params: SamplingParams object
            temperature: New temperature value
            top_p: New top_p value
            max_tokens: New max_tokens value
            
        Returns:
            Updated SamplingParams object
        """
        if not isinstance(sampling_params, SamplingParams):
            raise TypeError("sampling_params must be a vLLM SamplingParams object")
            
        params_dict = {
            "temperature": temperature if temperature is not None else sampling_params.temperature,
            "top_p": top_p if top_p is not None else sampling_params.top_p,
            "max_tokens": max_tokens if max_tokens is not None else sampling_params.max_tokens,
            "stop_token_ids": sampling_params.stop_token_ids
        }
        
        return SamplingParams(**params_dict)