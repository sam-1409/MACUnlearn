import random
import numpy as np
import torch
import os

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_model_state(model):
    return {k: v.cpu().clone() for k, v in model.state_dict().items()}

def set_model_state(model, state_dict):
    model.load_state_dict({k: v.to(model.device) for k, v in state_dict.items()})