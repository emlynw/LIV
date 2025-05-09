# liv/gcr_buffer.py
from collections import deque
import random
import torch

class GCRNegBuffer:
    """Stores negative (failed) goal frames for goal‑contrastive training."""
    def __init__(self, maxlen=50_000):
        self.buf = deque(maxlen=maxlen)

    def add_episode(self, traj_imgs, success: bool):
        """Optionally add a trajectory’s frames if the episode failed."""
        if success:
            return
        self.buf.extend(traj_imgs)          # keep *all* frames

    def sample(self, batch_size: int):
        batch_size = min(batch_size, len(self.buf))
        return random.sample(self.buf, batch_size)

    def __len__(self):
        return len(self.buf)
