import abc
import random
from abc import ABC
from enum import Enum

import nest_asyncio
import numpy as np
import torch
import torch.nn as nn
from torch import optim

from colabs.intro.api_endpoints import *

nest_asyncio.apply()

# Determine the best available device for PyTorch operations (Device Agnostic Code)
if torch.cuda.is_available():
    device = 'cuda'  # GPU
elif torch.backends.mps.is_available():
    device = 'mps'  # GPU for MacOS (Metal Programming Framework)
else:
    device = 'cpu'  # CPU

from collections import namedtuple, deque

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, state, action, next_state, reward):
        """Save a transition"""
        # Convert action to tensor if it's an integer
        if isinstance(action, int):
            action = torch.tensor([[action]], device=device, dtype=torch.long)
        self.memory.append(Transition(state, action, next_state, reward))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class SimpleNetwork(nn.Module):
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 64):
        super().__init__()
        self.layer1 = nn.Linear(state_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, action_size)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.layer3(nn.functional.relu(self.layer2(nn.functional.relu(self.layer1(state)))))


class DQL(abc.ABC):
    def __init__(self, n_observations, n_actions, params):
        self.steps_done = 0
        self.params = params
        self.n_actions = n_actions
        self.policy_net = SimpleNetwork(n_observations, n_actions, params.HIDDEN_SIZE).to(device)
        self.target_net = SimpleNetwork(n_observations, n_actions, params.HIDDEN_SIZE).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=params.LEARNING_RATE)
        self.memory = ReplayMemory(params.MEM_SIZE)

    def select_action(self, state: torch.Tensor) -> int:
        sample = random.random()
        eps_threshold = self.params.EPS_END + (self.params.EPS_START - self.params.EPS_END) * np.exp(
            -1. * self.steps_done / self.params.EPS_DECAY)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.inference_mode():
                # Get the action as an integer
                return self.policy_net(state.float()).max(1).indices.item()
        else:
            # Randomly select an action and return as integer
            return random.randrange(self.n_actions)

    def optimize_model(self):
        if len(self.memory) < self.params.BATCH_SIZE:
            return 0
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

    @abc.abstractmethod
    def train(self):
        raise NotImplementedError()


class SumoDQLSingleTLS(DQL, ABC):
    class Action(Enum):
        STEP = 0
        NEXT_PHASE = 1
        SWITCH_PROGRAM = 2

    def __init__(self, session_id: str, tls_id: str, n_observations, n_actions, params):
        super().__init__(n_observations, n_actions, params)
        self.session_id = session_id
        self.tls_id = tls_id

    def get_action_method(self, action):
        if action == self.Action.STEP:
            ret = lambda _: None
        elif action == self.Action.NEXT_PHASE:
            ret = lambda _: set_traffic_light_phase(self.tls_id, self.session_id,
                                                    make_step=0)  # Set next phase
        else:  # Switch Program
            selected_program_index = action - 2
            if selected_program_index >= len(selected_program_ids):
                raise RuntimeError("Illegal action")
            selected_program = selected_program_ids[int(selected_program_index)]
            ret = switch_traffic_light_program(tls_id=self.tls_id, session_id=self.session_id,
                                               program_id=selected_program,
                                               make_step=0)
        return ret

    def extract_state_tensor(self, response):
        is_ended = response['is_ended']
        metrics = response['vehicles_in_tls'][self.tls_id]['longest_waiting_time_car_in_lane']
        cars_that_left = response['cars_that_left']
        extracted_data = []
        for lane in metrics:
            values = list(metrics[lane].values())
            if values:
                extracted_data.extend([float(x) for x in metrics[lane].values()])
            else:
                extracted_data.extend([0. for _ in range(7)])
        state = torch.tensor(extracted_data, dtype=torch.float32, device=device)

        # print(metrics)
        reward = self.reward_func(metrics, cars_that_left)
        return state, reward, is_ended

    def reward_func(self, states: dict, cars_that_left: int) -> torch.Tensor:
        penalty = cars_that_left * 10
        for lane in states.values():
            if not lane:
                continue
            if lane.get('max_waiting_time', 0.) > 0:
                queue_length_percentage = lane['queue_length'] / (lane['total_cars'] / lane['occupancy'])
                penalty -= queue_length_percentage * lane['max_waiting_time']
            else:
                penalty += lane['average_speed']
        return torch.tensor(penalty, dtype=torch.float32, device=device)

    def train(self):
        pass
