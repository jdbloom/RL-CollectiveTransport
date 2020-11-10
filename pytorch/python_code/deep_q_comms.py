import os

import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class DeepQComms(nn.Module):
    def __init__(self, lr, num_actions, observation_size, num_ops_per_action):
        super().__init__()
