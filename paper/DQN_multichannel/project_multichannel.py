import sys
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential

import tensorflow as tf

with tf.Session() as sess:
    EPISODES = 8000000
    number_of_subset = 4
    Channel = np.zeros(16)
    def make_multiple_Channel(number_of_subset):
    
        numofele = int(16 / number_of_subset)
        Channel_order = np.array([[9, 10, 12, 14],[3, 8, 11, 13], [1, 2, 5, 6], [4, 7, 15, 16]])
        pop_Index = np.zeros(4)
        channel_list = np.array(range(1, 17), dtype='f8')
        for i in range(number_of_subset):
            # Channel_order[i] = np.random.choice(channel_list, numofele, replace=False)
    
            for j in range(numofele):
                pop_Index[j] = np.argwhere(channel_list == Channel_order[i][j])
    
            channel_list = np.delete(channel_list, pop_Index)
    
        Channel_order = Channel_order.astype('int32')
    
        return Channel_order
    
    def multiple_Channel(transition_prob, Channel_order, selected_subset, number_of_subset):
        # 2^n 개의 subset의 개수 선택
    
        numofele = 16/number_of_subset
    
        if sum(Channel) == 0:
            Channel[Channel_order[selected_subset]-1] = 1
    
        else:
            if random.random() < transition_prob:
    
    
                if selected_subset == numofele-1:
    
                    Channel[Channel_order[selected_subset] - 1] = 0
                    selected_subset = 0
                    Channel[Channel_order[selected_subset] - 1] = 1
    
                else:
    
                    Channel[Channel_order[selected_subset] - 1] = 0
                    selected_subset = selected_subset+1
                    Channel[Channel_order[selected_subset] - 1] = 1
    
        return Channel, selected_subset
    
    def single_good(transition_prob, start_channel):
    
        if sum(Channel) == 0:
    
            Channel[start_channel] = 1
    
        else:
    
            Info = np.argwhere(Channel == 1)
    
            if random.random() < transition_prob:
    
                if Info == [15]:
                    Channel[Info] = 0
                    Info = [0]
                    Channel[Info] = 1
    
                else:
                    Channel[Info] = 0
                    NextInfo = Info + [1]
                    Channel[NextInfo] = 1
    
        return Channel
    
    def Sensing_action(Select_channel, Channel):
    
        User_observation = np.zeros(len(Channel))
    
        if Channel[Select_channel] == 1:
    
            User_observation[Select_channel] = 1
    
        elif Channel[Select_channel] != 1:
    
            User_observation[Select_channel] = -1
    
        return User_observation
    
    class DQNAgent:
    
        def __init__(self, state_size, action_size):
    
            self.load_model = False
            # self.load_model = False
    
            # 상태와 행동의 크기 정의
            self.state_size = state_size
            self.action_size = action_size
    
            # DQN 하이퍼파라미터
            self.discount_factor = 0.9
            self.learning_rate = 0.0001
            self.epsilon = 1.0
            self.epsilon_decay = 0.999
            self.epsilon_min = 0.1
            self.batch_size = 32
            self.train_start = 32*50
            # 리플레이 메모리
            self.memory = deque(maxlen=1000000)
    
            # 모델과 타깃 모델 생성
            self.model = self.build_model()
            self.target_model = self.build_model()
    
            if self.load_model:
                self.model.load_weights("./save_model_comm/multi_DMA_dqn.h5")
    
            # 타깃 모델 초기화
            self.update_target_model()
    
        # 상태가 입력, 큐함수가 출력인 인공신경망 생성
        def build_model(self):
            model = Sequential()
            model.add(Dense(200, input_dim=self.state_size, activation='relu',
                            kernel_initializer='he_uniform'))
            model.add(Dense(200, activation='relu',
                            kernel_initializer='he_uniform'))
            model.add(Dense(self.action_size, activation='relu',
                            kernel_initializer='he_uniform'))
            model.summary()
            model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
            return model
    
        # 타깃 모델을 모델의 가중치로 업데이트
        def update_target_model(self):
            self.target_model.set_weights(self.model.get_weights())
    
        # 입실론 탐욕 정책으로 행동 선택
        def get_action(self, state):
    
    
            if np.random.rand() <= self.epsilon:
                return random.randrange(self.action_size)
            else:
                q_value = self.model.predict(state)
    
                return np.argmax(q_value)
    
        # 샘플 <s, a, r, s'>을 리플레이 메모리에 저장
        def append_sample(self, state, action, reward, next_state):
            self.memory.append((state, action, reward, next_state))
    
        # 리플레이 메모리에서 무작위로 추출한 배치로 모델 학습
        def train_model(self):
    
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
    
            # 메모리에서 배치 크기만큼 무작위로 샘플 추출
            mini_batch = random.sample(self.memory, self.batch_size)
    
            states = np.zeros((self.batch_size, self.state_size))
            next_states = np.zeros((self.batch_size, self.state_size))
            actions, rewards = [], []
    
            for i in range(self.batch_size):
                states[i] = mini_batch[i][0]
                actions.append(mini_batch[i][1])
                rewards.append(mini_batch[i][2])
                next_states[i] = mini_batch[i][3]
    
            # 현재 상태에 대한 모델의 큐함수
            # 다음 상태에 대한 타깃 모델의 큐함수
            target = self.model.predict(states) #state가 들어가면 action에 대한 Q-function 값이 나옴
            target_val = self.target_model.predict(next_states)
    
            # 벨만 최적 방정식을 이용한 업데이트 타깃
            for i in range(self.batch_size):
                target[i][actions[i]] = rewards[i] + self.discount_factor * (np.amax(target_val[i]))
    
            history = self.model.fit(states, target, batch_size=self.batch_size, epochs=1, verbose=0)
    
            return history
    
    
    
    if __name__ == "__main__":
    
    
        state_size = 256
        action_size = 16
        channel_size = 16
        transition_probability = 0.9
        agent = DQNAgent(state_size, action_size)
        actions, scores, episodes = [], [], []
        k = 0
        selected_subset = 3
    
        for e in range(EPISODES):
    
            if e == 0:
    
                for i in range(channel_size): # state 초기화 random select
    
                    Select_channel = random.randrange(channel_size)
                    channel_order = make_multiple_Channel(number_of_subset)
                    [channel, selected_subset] = multiple_Channel(0.8, channel_order, selected_subset, number_of_subset)
                    observation = Sensing_action(Select_channel, channel)
    
                    if i == 1:
    
                        state_nonreshape = np.vstack([past_observation, observation])
    
                    elif i == 0:
    
                        past_observation = observation
    
                    else:
                        state_nonreshape = np.vstack([state_nonreshape, observation])
    
    
            state = np.reshape(state_nonreshape, [1, state_size])
    
    
            # 현재 상태로 행동을 선택
            action = agent.get_action(state)
    
    
    
            # 선택한 행동으로 환경에서 한 타임스텝 진행
            [channel, selected_subset] = multiple_Channel(0.8, channel_order, selected_subset, number_of_subset)
            next_observation = Sensing_action(action, channel)
            current_state_nonreshape = state_nonreshape[1:(channel_size),0:(channel_size)]
            next_state_nonreshape = np.vstack([current_state_nonreshape,next_observation])
            next_state = np.reshape(next_state_nonreshape, [1, state_size])
    
            # 리플레이 메모리에 샘플 <s, a, r, s'> 저장
            reward = sum(next_observation)
            # reward = sum(sum(next_state))
            agent.append_sample(state, action, reward, next_state)
            # 매 타임스텝마다 학습
            #if reward == 1:
            #    print(reward)
    
            state = next_state
            state_nonreshape = next_state_nonreshape
            score = sum(sum(state))
    
            if len(agent.memory) >= agent.train_start:
                hist = agent.train_model()
    
            if e % 5000 == 0:
                # 에피소드 5000번에 한번씩 타깃 모델을 모델의 가중치로 업데이트
                agent.update_target_model()
    
            if 'hist' in locals():
                Loss_element = np.array(hist.history['loss'])
                #val_Loss_element = np.array(hist.history['val_loss'])
    
                if k == 0:
                    Loss = np.array(Loss_element)
                    #val_Loss = np.array(Loss_element)
                else:
                    Loss = np.hstack([Loss, Loss_element])
                    #val_Loss = np.hstack([Loss, Loss_element])
                k = k + 1
    
            # 에피소드마다 학습 결과 출력
            scores.append(score)
            episodes.append(e)
            actions.append(action)
    #append / pop 배열에 넣고 빼기
            #pylab.plot(episodes, scores, 'b')
            #pylab.savefig("./save_graph_comm/DMA_dqn.png")
    
            if e>10:
                action = 100 + actions[e]
                if score > 0:
                    print("episode:", e, "action", actions[e], "  score:", scores[e], "  memory length:", len(agent.memory),
                          "epsilon:", agent.epsilon)
                # if score > 6 and scores[9] != scores[8]:
                #     print("episode:", e-1, "action", actions[8], "  score:", scores[8], "  memory length:", len(agent.memory), "epsilon:", agent.epsilon)
                #     print("episode:", e, "action", actions[9], "  score:", scores[9], "  memory length:", len(agent.memory), "epsilon:", agent.epsilon)
    
    
    
        Numelement=np.unique(scores)
        total = np.zeros(len(Numelement))
    
        for i in range(len(Numelement)):
            total[i]=(scores==Numelement[i]).sum()
    
    
        plt.figure(1)
        plt.plot(Loss, 'b')
        plt.savefig("./save_graph_comm/multi_loss_dqn.png")
    
        plt.figure(2)
        plt.plot(episodes, scores, 'b')
        plt.savefig("./save_graph_comm/multi_DMA_dqn.png")
    
        plt.figure(3)
        plt.bar(Numelement, total)
        plt.savefig("./save_graph_comm/multi_distribution_dqn.png")
    
        # if agent.load_model == True:
        agent.model.save_weights("./save_model_comm/multi_DMA_dqn.h5")
        agent.model.save("./save_model_comm/multi_DMA_model_dqn.h5")
    
            #sys.exit()