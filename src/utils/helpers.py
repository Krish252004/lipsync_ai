import torch
import numpy as np
from typing import Union, Optional

def to_device(data: Union[torch.Tensor, dict, list], device: torch.device):
    """Move data to the specified device.
    
    Args:
        data: Input data (tensor, dict, or list)
        device: Target device
        
    Returns:
        Data moved to the target device
    """
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        return {k: to_device(v, device) for k, v in data.items()}
    elif isinstance(data, list):
        return [to_device(x, device) for x in data]
    else:
        raise TypeError(f"Unsupported type for device transfer: {type(data)}")