from __future__ import annotations

import numpy as np
import torch


class ReplayBuffer:
    """Packed replay buffer stored on the selected training device."""

    def __init__(self, state_dim, action_dim, capacity=100000, seed=0, device="cpu"):
        self.capacity = int(capacity)
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.device = torch.device(device)
        self.row_width = 2 * self.state_dim + self.action_dim + 2
        self.data = torch.empty(
            (self.capacity, self.row_width), dtype=torch.float32, device=self.device
        )
        self.position = 0
        self.size = 0
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(int(seed))

    def add(self, state, action, reward, next_state, done):
        # One packed transfer per transition when storage is on CUDA.
        row = np.empty(self.row_width, dtype=np.float32)
        offset = 0
        for value, width in (
            (state, self.state_dim),
            (action, self.action_dim),
            ([reward], 1),
            (next_state, self.state_dim),
            ([float(done)], 1),
        ):
            row[offset:offset + width] = value
            offset += width
        self.data[self.position].copy_(torch.as_tensor(row))
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def __len__(self):
        return self.size

    def sample(self, batch_size):
        idx = torch.randint(
            self.size, (int(batch_size),), generator=self.generator, device=self.device
        )
        batch = self.data.index_select(0, idx)
        s0 = self.state_dim
        a0 = s0 + self.action_dim
        r0 = a0 + 1
        ns0 = r0 + self.state_dim
        return (
            batch[:, :s0],
            batch[:, s0:a0],
            batch[:, a0:r0],
            batch[:, r0:ns0],
            batch[:, ns0:],
        )
