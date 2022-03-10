import numpy as np
import random
import time
import torch


def set_random_seed(seed):
    """Set random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_time_str():
    return time.strftime('%Y-%m-%d, %H:%M:%S', time.localtime())
