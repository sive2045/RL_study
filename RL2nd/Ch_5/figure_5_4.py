import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
current_path = os.path.dirname(os.path.abspath(__file__))
matplotlib.use('Agg')

ACTION_BACK = 0
ACTION_END = 1

# behavior policy
def behavior_policy():
    return np.random.binomial(1, 0.5)

# target policy
def target_policy():
    return ACTION_BACK

# one turn
def play():
    # track the action for importance ratio
    trajectory = []
    while True:
        action = behavior_policy()
        trajectory.append(action)
        if action == ACTION_END:
            return 0, trajectory
        if np.random.binomial(1, 0.9) == 0:
            return 1, trajectory

def figure_5_4():
    runs = 10
    episodes = 100000
    for run in range(runs):
        rewards = []
        for episode in range(0, episodes):
            reward, trajectory = play()
            if trajectory[-1] == ACTION_END:
                rho = 0
            else:
                rho = 1.0 / pow(0.5, len(trajectory))
            rewards.append(rho * reward)
        rewards = np.add.accumulate(rewards)
        estimations = np.asarray(rewards) / np.arange(1, episodes + 1)
        plt.plot(estimations)
    plt.xlabel('Episodes (log scale)')
    plt.ylabel('Ordinary Importance Sampling')
    plt.xscale('log')

    plt.savefig(current_path + '/images/figure_5_4.png')
    plt.close()

if __name__ == '__main__':
    figure_5_4()