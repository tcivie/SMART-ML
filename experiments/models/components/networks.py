from functools import reduce

import torch
import torch.nn as nn
import torch.nn.functional as F

from sumo_sim.Simulation import LightPhase


class SimpleNetwork(nn.Module):
    """
    A simple feedforward neural network with a single hidden layer.
    The network accepts input of size `state_size` and outputs a tensor of size `action_size`.
    The hidden layer has a size of `hidden_size` which is set to 64 by default and can be changed to a list of sizes.
    eg [64, 64] for two hidden layers of size 64 each.
    """

    def __init__(self, state_size: int, action_size: int, hidden_sizes: list[int]):
        super().__init__()
        self.intermediate_layer_results = None
        self.intermediate_layer_index = None
        self.action_size = action_size
        self.state_size = state_size
        assert hidden_sizes is not None, "Hidden sizes must be provided"
        assert len(hidden_sizes) > 0, "At least one hidden layer must be provided"

        self.layers = self.generate_layers(action_size, hidden_sizes, state_size)

    def set_intermediate_layer_to_extract(self, layer_index):
        self.intermediate_layer_index = layer_index
        self.intermediate_layer_results = None

    # noinspection PyListCreation
    @staticmethod
    def generate_layers(action_size, hidden_sizes, state_size):
        layers = []
        # Input layer
        layers.append(nn.Linear(state_size, hidden_sizes[0]))
        # Hidden layers
        for i in range(1, len(hidden_sizes)):
            layers.append(nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]))
        # Output layer
        layers.append(nn.Linear(hidden_sizes[-1], action_size))
        layers.append(nn.ReLU())
        if action_size > 1:
            layers.append(nn.Softmax(dim=0))
        return nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        for index, layer in enumerate(self.layers[:-1]):
            state = layer(state)
            if self.intermediate_layer_index is not None and index == self.intermediate_layer_index:
                self.intermediate_layer_results = state
            # state = F.relu(state)
        return self.layers[-1](state)


class SplitNetwork(nn.Module):
    """
    A feedforward neural network with a split architecture.
    The network accepts input of size `state_size`. The network decides at first step whether to skip the processing or if it should continue.
    In case it skips the network would return integer 0.
    In case it continues, the network would output a tensor of size `action_size`.

    The network would take as input for the second processing the results of one of the layers of the first processing.
    """

    def __init__(self, state_size: int, action_size: int, hidden_sizes: list[list[int]], feed_layer_index: int):
        """
        :param state_size: The size of the input tensor
        :param action_size: The size of the output tensor
        :param hidden_sizes: A list of hidden layer sizes for the first and second processing eg [[32,64], [64,32,64]]
        :param feed_layer_index: The index of the layer to feed to the second processing eg 1
        (the second layer [32,->64] would be fed to the second processing -> [64,32,64])
        """
        super().__init__()
        assert hidden_sizes is not None, "Hidden sizes must be provided"
        assert len(hidden_sizes) == 2, "Two sets of hidden layers must be provided"
        assert len(hidden_sizes[0]) > 0, "At least one hidden layer must be provided for the first set"
        assert len(hidden_sizes[1]) > 0, "At least one hidden layer must be provided for the second set"

        self.first_network = SimpleNetwork(state_size, 1, hidden_sizes[0])
        self.first_network.set_intermediate_layer_to_extract(feed_layer_index)
        self.second_network = SimpleNetwork(hidden_sizes[0][feed_layer_index], action_size, hidden_sizes[1])
        # Remove the sigmoid layer from the second network
        self.second_network.layers = self.second_network.layers[:-1]
        # Final layers for each controlled link with len(LightPhase) outputs
        self.final_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(action_size, len(LightPhase)),
                nn.ReLU(),
                nn.Softmax(dim=0)
            ) for _ in range(action_size // len(LightPhase))
        ])

        self.feed_layer_index = feed_layer_index
        self.state_size = state_size
        self.action_size = action_size

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        first_output = self.first_network(state)
        if first_output < 0.5:
            return first_output
        else:
            assert self.first_network.intermediate_layer_results is not None, "Intermediate layer results must be set"
            second_output = self.second_network(self.first_network.intermediate_layer_results)
            final_outputs = []
            for layer in self.final_layers:
                final_outputs.append(layer(second_output))
            final_outputs = torch.stack(final_outputs, dim=0)
            _, action_indices = final_outputs.max(dim=1)
            return action_indices


class LSTMNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMNetwork, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.state_size = input_size
        self.action_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def forward(self, x, h0, c0):
        out, (h0, c0) = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out, (h0, c0)
