from __future__ import annotations

import numpy as np
import torch


class ReplayBuffer:
    """Packed replay buffer stored on the selected training device.

    Optional auxiliary observations stay aligned with each transition. They are
    used by Gaussian rule priors that need the full raw environment state while
    the critic and actor consume agent-specific observations.
    """

    def __init__(self, state_dim, action_dim, capacity=100000, seed=0, device="cpu",
                 auxiliary_dim=0):
        self.capacity = int(capacity)
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.auxiliary_dim = int(auxiliary_dim)
        if self.auxiliary_dim < 0:
            raise ValueError("auxiliary_dim cannot be negative")
        self.device = torch.device(device)
        self.row_width = 2 * self.state_dim + self.action_dim + 2
        self.data = torch.empty(
            (self.capacity, self.row_width), dtype=torch.float32, device=self.device
        )
        self.auxiliary_data = (
            torch.empty(
                (self.capacity, 2 * self.auxiliary_dim),
                dtype=torch.float32,
                device=self.device,
            )
            if self.auxiliary_dim
            else None
        )
        self.position = 0
        self.size = 0
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(int(seed))

    def add(self, state, action, reward, next_state, done,
            auxiliary_state=None, next_auxiliary_state=None):
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
        if self.auxiliary_dim:
            if auxiliary_state is None or next_auxiliary_state is None:
                raise ValueError(
                    "This replay buffer requires current and next auxiliary observations"
                )
            auxiliary_row = np.empty(2 * self.auxiliary_dim, dtype=np.float32)
            auxiliary_row[:self.auxiliary_dim] = auxiliary_state
            auxiliary_row[self.auxiliary_dim:] = next_auxiliary_state
            self.auxiliary_data[self.position].copy_(torch.as_tensor(auxiliary_row))
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def __len__(self):
        return self.size

    def sample(self, batch_size):
        idx = self._sample_indices(batch_size)
        batch = self.data.index_select(0, idx)
        return self._unpack(batch)

    def sample_with_auxiliary(self, batch_size):
        if self.auxiliary_data is None:
            raise RuntimeError("This replay buffer has no auxiliary observations")
        idx = self._sample_indices(batch_size)
        batch = self.data.index_select(0, idx)
        auxiliary = self.auxiliary_data.index_select(0, idx)
        state, action, reward, next_state, done = self._unpack(batch)
        aux0 = self.auxiliary_dim
        return (
            state, action, reward, next_state, done,
            auxiliary[:, :aux0], auxiliary[:, aux0:],
        )

    def _sample_indices(self, batch_size):
        if self.size <= 0:
            raise ValueError("Cannot sample from an empty replay buffer")
        return torch.randint(
            self.size, (int(batch_size),), generator=self.generator, device=self.device
        )

    def _unpack(self, batch):
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
