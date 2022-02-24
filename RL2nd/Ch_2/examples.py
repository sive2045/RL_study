"""
Examples

Ex 2.5
Non-stationary problem example

Ex 2.11
Non-stationary parameter study 
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
    overloading Bandit class
    """
    def __init__(self, k_arm=10, epsilon=0., initial=0., step_size=0.1, sample_averages=False, UCB_param=None,
                 gradient=False, gradient_baseline=False, true_reward=0., exp_weighted=False):
        self.k = k_arm
        self.step_size = step_size
        self.sample_averages = sample_averages
        self.indices = np.arange(self.k)
        self.time = 0
        self.UCB_param = UCB_param
        self.gradient = gradient
        self.gradient_baseline = gradient_baseline
        self.average_reward = 0
        self.true_reward = true_reward
        self.epsilon = epsilon
        self.initial = initial
        self.exp_weighted = exp_weighted
        
    # take an action, update estimation for this action
    def step(self, action):
        # non-stationary
        self.q_true += np.random.randn(self.k)
        self.best_action = np.argmax(self.q_true)
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
        elif self.exp_weighted:
            self.q_estimation[action] += (reward - self.q_estimation[action]) * self.step_size
        else:
            # update estimation with constant step size
            self.q_estimation[action] += self.step_size * (reward - self.q_estimation[action])
        return reward

def simulate(runs, time, bandits):
    rewards = np.zeros((len(bandits), runs, time))
    best_action_counts = np.zeros(rewards.shape)
    for i, bandit in enumerate(bandits):
        for r in trange(runs):
            bandit.reset()
            for t in range(time):
                action = bandit.act()
                reward = bandit.step(action)
                rewards[i, r, t] = reward
                if action == bandit.best_action:
                    best_action_counts[i, r, t] = 1
    mean_best_action_counts = best_action_counts.mean(axis=1)
    mean_rewards = rewards.mean(axis=1)
    return mean_best_action_counts, mean_rewards


def figure_ex_2_5(runs=2000, time=10_000):
    # cal
    bandit = []
    bandit.append(BanditNonStationary(epsilon=0.1, exp_weighted=True))
    bandit.append(BanditNonStationary(epsilon=0.1, sample_averages=True))
    best_action_counts, avg_reward = simulate(runs, time, bandit)

    # plot
    plt.figure(figsize=(10, 20))

    plt.subplot(2, 1, 1)
    plt.plot(best_action_counts[0], label='Exponential Recency Weighted Meothd')
    plt.plot(best_action_counts[1], label='Samepl Average Method')
    plt.xlabel('Number of iterations'); plt.ylabel('% Optimal action')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(avg_reward[0], label='Exponential Recency Weighted Meothd')
    plt.plot(avg_reward[1], label='Samepl Average Method')
    plt.xlabel('Number of iterations'); plt.ylabel('% Average Reward')
    plt.legend()

    plt.savefig(current_path + '/images/figure_ex_2_5.png')
    plt.close()

def figure_ex_2_11(runs=2000, time=10_000):
    labels = ['epsilon-greedy', 'gradient bandit',
              'UCB', 'optimistic initialization', 'constant step size greedy']
    generators = [lambda epsilon: BanditNonStationary(epsilon=epsilon, sample_averages=True),
                  lambda alpha: BanditNonStationary(gradient=True, step_size=alpha, gradient_baseline=True),
                  lambda coef: BanditNonStationary(epsilon=0, UCB_param=coef, sample_averages=True),
                  lambda initial: BanditNonStationary(epsilon=0, initial=initial, step_size=0.1),
                  lambda exp: BanditNonStationary(epsilon=exp, exp_weighted=True)]
    parameters = [np.arange(-7, -1, dtype=np.float),
                  np.arange(-5, 2, dtype=np.float),
                  np.arange(-4, 3, dtype=np.float),
                  np.arange(-2, 3, dtype=np.float),
                  np.arange(-7, -1, dtype=np.float)]
    bandits = []
    for generator, parameter in zip(generators, parameters):
        for param in parameter:
            bandits.append(generator(pow(2, param)))

    _, average_rewards = simulate(runs, time, bandits)
    rewards = np.mean(average_rewards, axis=1)

    i = 0
    for label, parameter in zip(labels, parameters):
        l = len(parameter)
        plt.plot(parameter, rewards[i:i+l], label=label)
        i += l
    plt.xlabel('Parameter($2^x$)')
    plt.ylabel('Average reward')
    plt.legend()

    plt.savefig(current_path + '/images/figure_ex_2_11.png')
    plt.close()

if __name__ == '__main__':
    print("Start the processing....")
    figure_ex_2_5()
    figure_ex_2_11() # processing time : 310 min
    print('Processing completed')