from abc import ABC, abstractmethod

import torch


class BaseModel(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def select_action(self, old_state: torch.Tensor, current_state: torch.Tensor, reward) -> int:
        pass

    @abstractmethod
    def optimize_model(self):
        pass

