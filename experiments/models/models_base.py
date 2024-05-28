from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch


class BaseModel(ABC):

    @dataclass
    class Params:
        pass

    def __init__(self, params: 'BaseModel.Params'):
        self.params = params

    @abstractmethod
    def select_action(self, current_state: torch.Tensor, reward):
        raise NotImplementedError()

    @abstractmethod
    def optimize_model(self):
        raise NotImplementedError()


