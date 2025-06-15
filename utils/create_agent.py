from agents import RandomAgent, MinimaxAgent, APIAgent, VLLMAgent

try:
    from openai import OpenAI
    import httpx
except ImportError:
    print("WARNING: OpenAI Python package not installed. Install with 'pip install openai httpx'")
    OpenAI = None
    httpx = None
import os
import logging
import torch
from transformers import AutoTokenizer
try:
    from vllm import LLM, SamplingParams
except ImportError:
    logging.warning("vLLM not installed. To use vLLM backend, please install with 'pip install vllm'")
    LLM, SamplingParams = None, None
from config import Config

def create_api_agent(model, api_base_url, api_key, agent_name):
    """Create an API agent with the given configuration"""
    if OpenAI is None or httpx is None:
        raise ImportError("OpenAI and httpx packages are required for API agents")
    
    if not api_base_url or not api_key:
        raise ValueError(f"API base URL and API key are required for {agent_name}")
    
    try:
        client = OpenAI(
            base_url=api_base_url,
            api_key=api_key,
            http_client=httpx.Client(
                base_url=api_base_url,
                follow_redirects=True,
            ),
            timeout=httpx.Timeout(600, read=600, write=600, connect=600),
        )
        return APIAgent(
            api_client=client, 
            model=model,
            name=f"{agent_name}-{model}"
        )
    except Exception as e:
        raise RuntimeError(f"Failed to initialize {agent_name}: {e}")

def create_vllm_agent(model_path, agent_name):
    """Create a vLLM agent with the given configuration"""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        # Initialize vLLM engine
        llm_engine = LLM(
            model=model_path,
            tokenizer=model_path,
            tensor_parallel_size=Config.VLLM_TENSOR_PARALLEL_SIZE,
            trust_remote_code=True,
            max_num_seqs=Config.VLLM_MAX_NUM_SEQS,
            max_model_len=Config.VLLM_MAX_MODEL_LEN
        )
        
        # Create default sampling parameters
        sampling_params = SamplingParams(
            temperature=Config.TEMPERATURE,
            top_p=Config.TOP_P,
            max_tokens=Config.MAX_GENERATION_LENGTH,
            stop_token_ids=[tokenizer.eos_token_id] if tokenizer.eos_token_id else []
        )
        
        return VLLMAgent(llm_engine, sampling_params, tokenizer, name=agent_name, enable_thinking=Config.LOCAL_ENABLE_THINKING)
    except Exception as e:
        raise RuntimeError(f"Failed to initialize {agent_name}: {e}")

def create_agent(agent_type, agent_name, **kwargs):
    """Factory function to create agents based on type"""
    if agent_type == "api":
        return create_api_agent(
            kwargs.get('model', 'gpt-4-0125-preview'),
            kwargs.get('api_base_url'),
            kwargs.get('api_key'),
            agent_name
        )
    elif agent_type == "random":
        return RandomAgent(name=agent_name)
    elif agent_type == "minimax":
        return MinimaxAgent(name=agent_name)
    elif agent_type == "vllm":
        vllmAgent=create_vllm_agent(
            kwargs.get('model_path'),
            agent_name
        )
        if kwargs.get('enable_thinking')== False:
            vllmAgent.enable_thinking = False
        return vllmAgent
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")