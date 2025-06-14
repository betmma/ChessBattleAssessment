from agents import RandomAgent, MinimaxAgent, APIAgent, VLLMAgent
from utils.model_utils import ModelUtils

try:
    from openai import OpenAI
    import httpx
except ImportError:
    print("WARNING: OpenAI Python package not installed. Install with 'pip install openai httpx'")
    OpenAI = None
    httpx = None
    
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
        llm_engine, sampling_params, tokenizer = ModelUtils.initialize_vllm_model(
            model_path, 1
        )
        
        # Update sampling parameters
        sampling_params = ModelUtils.update_sampling_params(
            sampling_params,
            temperature=0.7,
            top_p=0.95,
            max_tokens=1024
        )
        
        return VLLMAgent(llm_engine, sampling_params, tokenizer, name=agent_name, enable_thinking=False)
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
        return create_vllm_agent(
            kwargs.get('model_path'),
            agent_name
        )
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")