import json
import numpy as np
def clean_np_types(obj):
    """
    Recursively converts numpy types to native Python types.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, dict):
        return {k: clean_np_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_np_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(clean_np_types(item) for item in obj)
    else:
        return obj
def safe_json_dump(obj, *args, **kwargs):
    cleaned_data = clean_np_types(obj)
    json.dump(cleaned_data, *args, **kwargs)
