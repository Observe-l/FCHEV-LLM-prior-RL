from __future__ import annotations

import numpy as np


class ReplayBuffer:
    def __init__(self, state_dim, action_dim, capacity=100000, seed=0):
        self.capacity = int(capacity)
        self.state = np.zeros((capacity, state_dim), dtype=np.float32)
        self.action = np.zeros((capacity, action_dim), dtype=np.float32)
        self.reward = np.zeros((capacity, 1), dtype=np.float32)
        self.next_state = np.zeros((capacity, state_dim), dtype=np.float32)
        self.done = np.zeros((capacity, 1), dtype=np.float32)
        self.position = 0
        self.size = 0
        self.rng = np.random.default_rng(seed)

    def add(self, state, action, reward, next_state, done):
        i = self.position
        self.state[i] = state
        self.action[i] = action
        self.reward[i, 0] = reward
        self.next_state[i] = next_state
        self.done[i, 0] = float(done)
        self.position = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def __len__(self):
        return self.size

    def sample(self, batch_size):
        idx = self.rng.integers(0, self.size, size=int(batch_size))
        return self.state[idx], self.action[idx], self.reward[idx], self.next_state[idx], self.done[idx]
