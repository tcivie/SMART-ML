import random
from collections import deque, namedtuple

from experiments import torch, device

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
        self.old_state = None

    def push(self, action, new_state, reward):
        """Save a transition"""
        if len(self.memory) == 0:
            self.old_state = new_state
            return
        # Convert action to tensor if it's an integer
        if isinstance(action, int):
            action = torch.tensor([[action]], device=device, dtype=torch.long)
        self.memory.append(Transition(self.old_state, action, new_state, reward))
        self.old_state = new_state

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
