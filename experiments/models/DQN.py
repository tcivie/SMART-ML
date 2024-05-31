import abc
import random
from collections import namedtuple
from dataclasses import dataclass
from typing import Union

import numpy as np
import torch.optim
from overrides import overrides
from torch import nn

from experiments import device
from experiments.models.components.memory import ReplayMemory
from experiments.models.components.networks import SimpleNetwork
from experiments.models.models_base import BaseModel
from sumo_sim.Simulation import LightPhase

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class DQN(BaseModel):
    @dataclass
    class Params(BaseModel.Params):
        observations: int = 7
        policy_net: nn.Module = SimpleNetwork(7, 3, [64, 64])
        target_net: nn.Module = SimpleNetwork(7, 3, [64, 64])
        optimizer: torch.optim.Optimizer = torch.optim.Adam
        memory: ReplayMemory = ReplayMemory(10000)

        EPS_START: float = 0.9
        EPS_END: float = 0.05
        EPS_DECAY: float = 200

        GAMMA: float = 0.999
        BATCH_SIZE: int = 128
        TARGET_UPDATE: int = 10

    def __init__(self, params: 'DQN.Params'):
        super().__init__(params)
        self.steps_done = 0
        self.actions = params.policy_net.action_size
        self.policy_net = params.policy_net.to(device)
        self.target_net = params.target_net.to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = params.optimizer
        self.memory = params.memory

    def __str__(self):
        return f"DQN: {self.params}"

    def __repr__(self):
        return f"DQN: {self.params}"

    def select_action(self, current_state: torch.Tensor, reward):
        sample = random.random()
        eps_threshold = self.params.EPS_END + (self.params.EPS_START - self.params.EPS_END) * np.exp(
            -1. * self.steps_done / self.params.EPS_DECAY)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.inference_mode():
                action = self.policy_net(current_state.float()).max(0).indices.item()
                self.memory.push(action, current_state, reward)
                return action
        else:
            # Randomly select an action and return as integer
            action = random.randrange(self.actions)
            self.memory.push(action, current_state, reward)
            return action

    def optimize_model(self):
        if len(self.memory) < self.params.BATCH_SIZE:
            return 0

        if self.steps_done % self.params.TARGET_UPDATE == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        transitions = self.memory.sample(self.params.BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action).to(torch.int64)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1).values
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.params.BATCH_SIZE, device=device)
        with torch.inference_mode():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.params.GAMMA) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 1000)
        self.optimizer.step()
        return loss


class SplitDQN(DQN):
    @dataclass
    class Params(DQN.Params):
        num_of_controlled_links: int = 1

    def __init__(self, params: 'SplitDQN.Params'):
        super().__init__(params)
        self.num_of_controlled_links = params.num_of_controlled_links

    @overrides
    def select_action(self, current_state: torch.Tensor, reward):
        sample = random.random()
        eps_threshold = self.params.EPS_END + (self.params.EPS_START - self.params.EPS_END) * np.exp(
            -1. * self.steps_done / self.params.EPS_DECAY)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.inference_mode():
                action = self.policy_net(current_state.float())
                action = [LightPhase(a.item()) for a in action]
                self.memory.push(action, current_state, reward)
                return action
        else:
            # Randomly select an action and return as integer
            if random.random() < 0.5:  # Skip action and return 0
                self.memory.push(0, current_state, reward)
                return 0
            else:
                action = [random.choice(list(LightPhase)) for _ in range(self.num_of_controlled_links)]
                self.memory.push(action, current_state, reward)
                return action
