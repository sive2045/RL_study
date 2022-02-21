"""
Ex 2.5
Non-stationary problem example
"""
from Bandit import Bandit
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange
import os
current_path = os.path.dirname(os.path.abspath(__file__))

matplotlib.use('Agg')

class BanditNonStationary(Bandit):
    """
    overide Bandit class
    """
    def __init__(self, non_stationary=False):
        self.non_stationary = non_stationary
        
    # take an action, update estimation for this action
    def step(self, action):
        # generate the reward under N(real reward, 1)
        reward = np.random.randn() + self.q_true[action]
        self.time += 1
        self.action_count[action] += 1
        self.average_reward += (reward - self.average_reward) / self.time

        if self.sample_averages:
            # update estimation using sample averages
            self.q_estimation[action] += (reward - self.q_estimation[action]) / self.action_count[action]
        elif self.gradient:
            one_hot = np.zeros(self.k)
            one_hot[action] = 1
            if self.gradient_baseline:
                baseline = self.average_reward
            else:
                baseline = 0
            self.q_estimation += self.step_size * (reward - baseline) * (one_hot - self.action_prob)
        elif self.non_stationary:
            self.q_estimation[action] += (reward - self.q_estimation[action]) / self.step_size
        else:
            # update estimation with constant step size
            self.q_estimation[action] += self.step_size * (reward - self.q_estimation[action])
        return reward