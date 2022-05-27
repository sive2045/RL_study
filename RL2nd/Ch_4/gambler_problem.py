import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
current_path = os.path.dirname(os.path.abspath(__file__))
matplotlib.use('Agg')

"""
Ex. 4.9
Gambler's Problem

0. Probability of head: 0.10
1. Probability of head: 0.25
2. Probability of head: 0.55
3. Probability of head: 0.50

state: 1, 2, ... , 99 $
actions: betting the money $ 0, 1, ,,, min(s, 100 - s) 
reward: +1 if haed else 0
"""

# 1$ ~ MAX BUDGET
GOAL = 100

STATES = np.arange(GOAL +1)

def step(state, action):
    if state < 1 or state > 99:
        pass

def sol_ex_4_3(prob_h):
    state_value = np.zeros(GOAL + 1)
    state_value[GOAL] = 1.0
    sweep_history = []
    iteration = 0

    # value iteration
    while True:
        old_state_value = state_value.copy()
        sweep_history.append(old_state_value)

        for state in STATES[1:GOAL]:
            actions = np.arange(min(state, GOAL - state) + 1)
            action_returns = []
            for action in actions:                
                action_returns.append(
                    prob_h * state_value[state + action] + (1 - prob_h) * state_value[state - action]
                )
            new_value = np.max(action_returns)
            state_value[state] = new_value
        
        max_delta_value = abs(old_state_value - state_value).max()
        print(f'iterations : {iteration}, max delta value : {max_delta_value}')
        if max_delta_value < 1e-9:
            sweep_history.append(state_value)
            break
        iteration += 1

    # compute the optimal policy
    policy = np.arange(GOAL + 1)
    for state in STATES[1:GOAL]:
        actions = np.arange(min(state, GOAL - state) + 1)
        action_returns = []
        for action in actions:
            action_returns.append(
                prob_h * state_value[state + action] + (1 - prob_h) * state_value[state - action]
            )
        policy[state] = actions[np.argmax(np.round(action_returns[1:], 5)) + 1]
        print(f'policy{state}: {policy[state]}')
    
    print('ploting...')
    plt.figure(figsize=(10, 20))
    plt.subplot(2, 1, 1)
    for sweep, state_value in enumerate(sweep_history):
        plt.plot(state_value, label='sweep {}'.format(sweep))
    plt.xlabel('Capital')
    plt.ylabel('Value estimates')
    plt.legend(loc='best')
    plt.subplot(2, 1, 2)
    plt.scatter(STATES, policy)
    plt.xlabel('Capital')
    plt.ylabel('Final policy (stake)')
    plt.grid()
    plt.savefig(current_path + '/images/figure_4_3_50.png')
    plt.close()
    print('done!')

if __name__ == '__main__':
    #sol_ex_4_3(0.25)
    #sol_ex_4_3(0.55)
    #sol_ex_4_3(0.10)
    sol_ex_4_3(0.50)