import numpy as np
import torch as T


def choose_action(observation, epsilon, test):
    if test or np.random.random() > 