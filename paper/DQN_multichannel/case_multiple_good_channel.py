#######################################################################
# Copyright (C)                                                       #
# 2022 Chungneung Lee(lc9902130509@gmail.com)                         #
# Released under the MIT license.                                     #
#######################################################################

import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from keras.models import Sequential

EPISODES = 500_000
CHANNEL = 10 * np.ones(16) # init channel

def make_multiple_good_channel_model(switching_prob=0.9, channel_size=16, time_slot=50):
    """
    channel_size 변화시 코드 수정 필요
    """
    channels = np.zeros((time_slot, channel_size))
    fixed_channel_order = np.array([[8, 9, 11, 13],[2, 7, 10, 12], [0, 1, 4, 5], [3, 6, 14, 15]])

    idx_order = 0
    for time in range(time_slot):
        channels[time][fixed_channel_order[idx_order]] = 1

        if np.random.binomial(1, switching_prob) == 1:
            idx_order = idx_order + 1 if idx_order < 3 else 0
    
    print(f'Channel Set: \n{channels}')
    return channels

def sensing_action(select_channel_idx, channel):
    user_observation = np.zeros(len(channel))

    if channel[select_channel_idx] == 1:
        user_observation[select_channel_idx] = 1
    else:
        user_observation[select_channel_idx] = -1
    
    return user_observation