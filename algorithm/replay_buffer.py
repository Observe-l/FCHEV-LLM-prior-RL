from __future__ import annotations

import numpy as np
import torch


class ReplayBuffer:
    """Packed replay buffer stored on the selected training device."""

    def __init__(self, state_dim, action_dim, capacity=100000, seed=0, device="cpu",
                 extra_dim=0):
        self.capacity = int(capacity)
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.extra_dim = int(extra_dim)
        if self.extra_dim < 0:
            raise ValueError("extra_dim cannot be negative")
        self.device = torch.device(device)
        self.row_width = 2 * self.state_dim + self.action_dim + 2
        self.data = torch.empty(
            (self.capacity, self.row_width), dtype=torch.float32, device=self.device
        )
        self.extra = (
            torch.empty((self.capacity, self.extra_dim), dtype=torch.float32, device=self.device)
            if self.extra_dim else None
        )
        self.position = 0
        self.size = 0
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(int(seed))

    def add(self, state, action, reward, next_state, done, extra=None):
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
        if self.extra_dim:
            if extra is None:
                extra = np.zeros(self.extra_dim, dtype=np.float32)
            extra = np.asarray(extra, dtype=np.float32).reshape(-1)
            if extra.shape != (self.extra_dim,) or not np.all(np.isfinite(extra)):
                raise ValueError(f"extra metadata must be finite with shape ({self.extra_dim},)")
            self.extra[self.position].copy_(torch.as_tensor(extra))
        elif extra is not None:
            raise ValueError("This replay buffer was created without extra metadata")
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
        result = (
            batch[:, :s0],
            batch[:, s0:a0],
            batch[:, a0:r0],
            batch[:, r0:ns0],
            batch[:, ns0:],
        )
        if self.extra_dim:
            extra_batch = self.extra.index_select(0, idx)
            return (*result, extra_batch)
        return result
