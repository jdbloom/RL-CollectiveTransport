# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 20:09:08 2020

@author: aaaaambition
"""
import numpy as np

class Agent:
    def __init__(self):
        pass


    def make_action(self, observation=None, learn=False):
        '''
        If no observation is supplied, it uses the most recent one in memory
        '''
        x = np.array([-1,-1,-10],dtype='float')
        return x

    def receive_observation(self, observation):
        pass
