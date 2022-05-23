#######################################################################
# Copyright (C)                                                       #
# 2022 Chungneung Lee(lc9902130509@gmail.com)                         #
# Released under the MIT license.                                     #
#######################################################################

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from collections import deque
from keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from keras.models import Sequential
import os
current_path = os.path.dirname(os.path.abspath(__file__))

def make_multiple_good_channel_model(switching_prob=0.9, channel_size=16, time_slot=50):
    """
    channel_size, # of subset 변화시 코드 수정 필요
    (iid channel 예제이므로 무시해도 무관)
    """
    channels = np.zeros((time_slot, channel_size))
    fixed_channel_order = np.array([[8, 9, 11, 13],[2, 7, 10, 12], [0, 1, 4, 5], [3, 6, 14, 15]])

    idx_order = 0
    for time in range(time_slot):
        channels[time][fixed_channel_order[idx_order]] = 1

        if np.random.binomial(1, switching_prob) == 1:
            idx_order = idx_order + 1 if idx_order < 3 else 0
    
    #print(f'Channel Set: \n{channels}')
    return channels

def sensing_action(select_channel_idx, channel):
    user_observation = np.zeros(len(channel))

    if channel[select_channel_idx] == 1:
        user_observation[select_channel_idx] = 1
    else:
        user_observation[select_channel_idx] = -1
    
    return user_observation

class DQN:
    """
    DQN model class
    """
    def __init__(self, state_size, action_size, load_model=False, discount_factor=0.9, learning_rate=1e-4, \
        epsilon_min=0.1, batch_size=32, train_start=32*50, replay_size=1_000_000):
        
        self.load_model = load_model
        if self.load_model:
            self.model.load_weights(current_path + "./data/models/multi_DMA_dqn.h5")
        
        self.state_size = state_size
        self.action_size = action_size
        
        # DQN Hyperparameters (in papaer) 
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        self.epsilon = 1.0
        self.epsilon_decay = 0.999
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.train_start = train_start
        
        # replay memory
        self.memory = deque(maxlen=replay_size)
        
        # build model and target model
        self.model = self.build_model()
        self.target_model = self.build_model()
        
        # init target model
        self.update_target_model()
    
    
    def build_model(self):
        """
        input: state, output: q-function
        """
        model = Sequential()
        model.add(Dense(50, input_dim=self.state_size, activation='relu',
                        kernel_initializer='he_uniform'))
        model.add(Dense(50, activation='relu',
                        kernel_initializer='he_uniform'))
        model.add(Dense(50, activation='relu',
                        kernel_initializer='he_uniform'))
        model.add(Dense(self.action_size, activation='relu',
                        kernel_initializer='he_uniform'))
        model.summary()
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model
    
    def update_target_model(self):
        """
        update targets weights from model weights
        """
        self.target_model.set_weights(self.model.get_weights())
        
    def get_action(self, state):
        """
        follow e-greedy method
        """
        if np.random.rand() <= self.epsilon:
            return np.random.random_integers(self.action_size)-1
        else:
            q_value = self.model.predict(state)
            
            return np.argmax(q_value)
    
    def append_sample(self, state, action, reward, next_state):
        """
        <s, a, r, s'> store in replay memory
        """
        self.memory.append((state, action, reward, next_state))
    
    def train_model(self):
        """
        learn model from data that
        random sampling from replay memory
        """
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        mini_batch = random.sample(self.memory, self.batch_size)
        
        states = np.zeros((self.batch_size, self.state_size))
        next_states = np.zeros((self.batch_size, self.state_size))
        actions, rewards = [], []

        for i in range(self.batch_size):
            states[i] = mini_batch[i][0]
            actions.append(mini_batch[i][1])
            rewards.append(mini_batch[i][2])
            next_states[i] = mini_batch[i][3]

        # present state's q-function
        target = self.model.predict(states) 
        # next state's q-function 
        target_val = self.target_model.predict(next_states)

        # the Bellman Eqs
        for i in range(self.batch_size):
            target[i][actions[i]] = rewards[i] + self.discount_factor * (np.amax(target_val[i]))

        history = self.model.fit(states, target, batch_size=self.batch_size, epochs=1, verbose=0)

        return history

if __name__ == '__main__':
    EPISODES = 1_000_000
    state_size = 256
    action_szie = 16
    channel_size = 16
    agent = DQN(state_size, action_szie)
    
    actions, scores, episodes = [], [] , []
    k = 0
    start_channel = 7
    
    raw_data = pd.read_csv(current_path + './data/real_data_trace.csv')
    raw_data = raw_data.drop('index', axis=1)
    data =raw_data.to_numpy()
    
    for ep in range(EPISODES):
        if ep == 0:
            for i in range(channel_size):
                if i == 15:
                    e = 15
                
                select_channel = random.randrange(channel_size)
                channel = data[i]
                observation_channel = sensing_action(select_channel, channel)
                
                if i == 0:
                    past_observation_channel = observation_channel
                elif i == 1:
                    state_nonreshape = np.vstack([past_observation_channel, observation_channel])
                else:
                    state_nonreshape = np.vstack([state_nonreshape, observation_channel])
    
        state = np.reshape(state_nonreshape, [1, state_size])

        # e-greedy action methods
        action = agent.get_action(state)

        data_index = ep % len(data)
        channel = data[data_index]
        next_observation = sensing_action(action, channel)
        current_state_nonreshape = state_nonreshape[1:(channel_size),0:(channel_size)]
        next_state_nonreshape = np.vstack([current_state_nonreshape, next_observation])
        next_state = np.reshape(next_state_nonreshape, [1, state_size])
        
        reward = sum(next_observation)
        
        agent.append_sample(state, action, reward, next_state)
        
        state = next_state
        state_nonreshape = next_state_nonreshape
        score = sum(sum(state))
        
        if len(agent.memory) >= agent.train_start:
                hist = agent.train_model()
    
        if ep % 5_000 == 0:
            agent.update_target_model()

        if 'hist' in locals():
            loss_element = np.array(hist.history['loss'])
            #val_loss_element = np.array(hist.history['val_loss'])
            if k == 0:
                loss = np.array(loss_element)
            else:
                loss = np.hstack([loss, loss_element])
                #val_loss = np.hstack([loss, loss_element])
            k = k + 1 
        
        # learing results per ep
        scores.append(score)
        episodes.append(ep)
        actions.append(action)
        
        if ep>10:
            action = 100 + actions[ep]
            if score > 0:
                print(f"episode: {ep}, action: {actions[ep]}, score: {scores[ep]}\
                     memory lenght: {len(agent.memory)}")
    
    numelement=np.unique(scores)
    total = np.zeros(len(numelement))

    for i in range(len(numelement)):
        total[i]=(scores==numelement[i]).sum()


    plt.figure(1)
    plt.plot(loss, 'b')
    plt.savefig(current_path + "./data/images/multi_loss_dqn.png")

    plt.figure(2)
    plt.plot(episodes, scores, 'b')
    plt.savefig(current_path + "./data/images/multi_DMA_dqn.png")

    plt.figure(3)
    plt.bar(numelement, total)
    plt.savefig(current_path + "./data/images/multi_distribution_dqn.png")

    agent.model.save_weights(current_path + "./data/models/multi_DMA_dqn.h5")
    agent.model.save(current_path + "./data/models/multi_DMA_model_dqn.h5")