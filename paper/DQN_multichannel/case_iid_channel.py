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
CHANNEL = 100 * np.ones(16)
threshold_prob = 0.5

def iid_channel(channels ,transition_prob):
    pass