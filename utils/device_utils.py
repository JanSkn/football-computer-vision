import torch

def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"  
    elif torch.backends.mps.is_available():
        return "mps"  # Metal Performance Shaders (Apple Silicon)
    else:
        return "cpu"