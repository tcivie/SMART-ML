from abc import ABC, abstractmethod

import torch


class BaseModel(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def select_action(self, current_state: torch.Tensor, reward) -> int:
        raise NotImplementedError()

    @abstractmethod
    def optimize_model(self):
        raise NotImplementedError()


