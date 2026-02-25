import pdb

import numpy as np
import collections
import random
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action_dict, reward, next_state, done):
        self.buffer.append((state, action_dict, reward, next_state, done))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        state, action_dict, reward, next_state, done = zip(*transitions)
        return np.array(state), action_dict, reward, np.array(next_state), done

    def size(self):
        return len(self.buffer)
