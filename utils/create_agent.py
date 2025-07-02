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

def create_api_agent(model, api_base_url, api_key):
    """Create an API agent with the given configuration"""
    if OpenAI is None or httpx is None:
        raise ImportError("OpenAI and httpx packages are required for API agents")
    
    if not api_base_url or not api_key:
        raise ValueError(f"API base URL and API key are required for {model}")
    
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
        )
    except Exception as e:
        raise RuntimeError(f"Failed to initialize api {model}: {e}")

def create_vllm_agent(model_path):
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
            max_num_batched_tokens=Config.VLLM_MAX_NUM_BATCHED_TOKENS,
            max_num_seqs=Config.VLLM_MAX_NUM_SEQS,
            max_model_len=Config.VLLM_MAX_MODEL_LEN,
            gpu_memory_utilization=Config.VLLM_GPU_MEMORY_UTILIZATION,
        )
        
        # Create default sampling parameters
        sampling_params = SamplingParams(
            temperature=Config.TEMPERATURE,
            top_p=Config.TOP_P,
            max_tokens=Config.MAX_GENERATION_LENGTH,
            stop_token_ids=[tokenizer.eos_token_id] if tokenizer.eos_token_id else []
        )
        
        return VLLMAgent(llm_engine, sampling_params, tokenizer, enable_thinking=Config.LOCAL_ENABLE_THINKING)
    except Exception as e:
        raise RuntimeError(f"Failed to initialize vllm {model_path}: {e}")

def create_agent(agent_type, agent_name=None, **kwargs):
    """Factory function to create agents based on type"""
    if agent_type == "api":
        agent = create_api_agent(
            kwargs.get('model', 'gpt-4-0125-preview'),
            kwargs.get('api_base_url'),
            kwargs.get('api_key')
        )
    elif agent_type == "random":
        agent = RandomAgent()
    elif agent_type == "minimax":
        agent = MinimaxAgent(random_chance=kwargs.get('random_chance', 0.0))
    elif agent_type == "vllm":
        agent=create_vllm_agent(
            kwargs.get('model_path'),
        )
        if kwargs.get('enable_thinking')== False:
            agent.enable_thinking = False
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
    if agent_name:
        agent.name = agent_name
    return agent