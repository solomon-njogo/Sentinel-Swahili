"""
Device Utilities for Transformer Training
Handles device detection, GPU management, and tensor movement.
"""

import logging
import torch
from typing import Optional, Any, Union

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_device(use_cuda: bool = True, device: Optional[str] = None) -> torch.device:
    """
    Get the appropriate device for training.
    
    Args:
        use_cuda: Whether to use CUDA if available (default: True)
        device: Specific device to use ('cuda', 'cpu', 'cuda:0', etc.) or None for auto-detect
    
    Returns:
        torch.device instance
    """
    if device is not None:
        # Use specified device
        device_obj = torch.device(device)
        logger.info(f"Using specified device: {device_obj}")
        return device_obj
    
    if use_cuda and torch.cuda.is_available():
        device_obj = torch.device('cuda')
        logger.info(f"Using CUDA device: {device_obj}")
        logger.info(f"CUDA device name: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA device count: {torch.cuda.device_count()}")
    else:
        device_obj = torch.device('cpu')
        if use_cuda:
            logger.warning("CUDA requested but not available, using CPU")
        else:
            logger.info("Using CPU device")
    
    return device_obj


def move_to_device(
    obj: Any,
    device: Union[torch.device, str]
) -> Any:
    """
    Move an object (tensor, model, or dict/list of tensors) to the specified device.
    
    Args:
        obj: Object to move (tensor, model, dict, list, etc.)
        device: Target device
    
    Returns:
        Object moved to device
    """
    if isinstance(device, str):
        device = torch.device(device)
    
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, torch.nn.Module):
        return obj.to(device)
    elif isinstance(obj, dict):
        return {key: move_to_device(value, device) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return type(obj)(move_to_device(item, device) for item in obj)
    else:
        # Return as-is if not a tensor or model
        return obj


def get_device_info() -> dict:
    """
    Get information about available devices.
    
    Returns:
        Dictionary with device information
    """
    info = {
        'cuda_available': torch.cuda.is_available(),
        'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'current_device': None,
        'device_name': None,
        'cuda_version': None,
        'cudnn_version': None,
    }
    
    if torch.cuda.is_available():
        info['current_device'] = torch.cuda.current_device()
        info['device_name'] = torch.cuda.get_device_name(0)
        info['cuda_version'] = torch.version.cuda
        info['cudnn_version'] = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
        
        # Memory information
        info['cuda_memory_allocated'] = torch.cuda.memory_allocated(0) / (1024 ** 3)  # GB
        info['cuda_memory_reserved'] = torch.cuda.memory_reserved(0) / (1024 ** 3)  # GB
        info['cuda_max_memory_allocated'] = torch.cuda.max_memory_allocated(0) / (1024 ** 3)  # GB
    
    return info


def print_device_info():
    """Print device information to console."""
    info = get_device_info()
    
    print("=" * 80)
    print("Device Information")
    print("=" * 80)
    print(f"CUDA Available: {info['cuda_available']}")
    print(f"CUDA Device Count: {info['cuda_device_count']}")
    
    if info['cuda_available']:
        print(f"Current Device: {info['current_device']}")
        print(f"Device Name: {info['device_name']}")
        print(f"CUDA Version: {info['cuda_version']}")
        if info['cudnn_version']:
            print(f"cuDNN Version: {info['cudnn_version']}")
        print(f"Memory Allocated: {info['cuda_memory_allocated']:.2f} GB")
        print(f"Memory Reserved: {info['cuda_memory_reserved']:.2f} GB")
        print(f"Max Memory Allocated: {info['cuda_max_memory_allocated']:.2f} GB")
    else:
        print("Using CPU")
    print("=" * 80)


def clear_gpu_cache():
    """Clear GPU cache to free up memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("GPU cache cleared")
    else:
        logger.warning("CUDA not available, nothing to clear")


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # For reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    logger.info(f"Random seed set to {seed}")


if __name__ == "__main__":
    # Test device utilities
    print("=" * 80)
    print("Testing Device Utilities")
    print("=" * 80)
    
    # Get device
    device = get_device()
    print(f"\nSelected device: {device}")
    
    # Print device info
    print_device_info()
    
    # Test moving tensors
    print("\n" + "=" * 80)
    print("Testing Tensor Movement")
    print("=" * 80)
    
    tensor = torch.randn(3, 4)
    print(f"Original tensor device: {tensor.device}")
    
    moved_tensor = move_to_device(tensor, device)
    print(f"Moved tensor device: {moved_tensor.device}")
    
    # Test moving dict
    data_dict = {
        'input_ids': torch.randn(2, 10),
        'attention_mask': torch.ones(2, 10),
        'labels': torch.randint(0, 3, (2,))
    }
    
    print(f"\nOriginal dict devices:")
    for key, value in data_dict.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.device}")
    
    moved_dict = move_to_device(data_dict, device)
    print(f"\nMoved dict devices:")
    for key, value in moved_dict.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.device}")
    
    # Test seed setting
    print("\n" + "=" * 80)
    print("Testing Seed Setting")
    print("=" * 80)
    set_seed(42)
    tensor1 = torch.randn(3, 3)
    set_seed(42)
    tensor2 = torch.randn(3, 3)
    print(f"Tensors are equal: {torch.equal(tensor1, tensor2)}")
    
    print("\n" + "=" * 80)
    print("Device Utilities Test Complete!")
    print("=" * 80)

