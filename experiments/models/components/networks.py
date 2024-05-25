from functools import reduce

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleNetwork(nn.Module):
    """
    A simple feedforward neural network with a single hidden layer.
    The network accepts input of size `state_size` and outputs a tensor of size `action_size`.
    The hidden layer has a size of `hidden_size` which is set to 64 by default and can be changed to a list of sizes.
    eg [64, 64] for two hidden layers of size 64 each.
    """
    def __init__(self, state_size: int, action_size: int, hidden_sizes: list[int]):
        super().__init__()
        assert hidden_sizes is not None, "Hidden sizes must be provided"
        assert len(hidden_sizes) > 0, "At least one hidden layer must be provided"

        layers = []

        # Input layer
        layers.append(nn.Linear(state_size, hidden_sizes[0]))

        # Hidden layers
        for i in range(1, len(hidden_sizes)):
            layers.append(nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]))

        # Output layer
        layers.append(nn.Linear(hidden_sizes[-1], action_size))
        layers.append(nn.Sigmoid())

        self.layers = nn.ModuleList(layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        for layer in self.layers[:-1]:
            state = F.relu(layer(state))
        return self.layers[-1](state)

