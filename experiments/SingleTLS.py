from abc import ABC
from enum import Enum

import torch

from api_endpoints import get_initial_data, set_traffic_light_phase, switch_traffic_light_program
from experiments import device
from experiments.experiments_base import Experiment
from experiments.models.DQN import DQN
from experiments.models.models_base import BaseModel


class SumoSingleTLSExperiment(Experiment):
    class Action(Enum):
        STEP = 0
        NEXT_PHASE = 1
        SWITCH_PROGRAM = 2

    def __init__(self, session_id: str, tls_id: str, model: BaseModel, reward_func=None):
        super().__init__(get_initial_data(session_id))
        if reward_func is None:
            self.reward_func = self.default_reward_func
        else:
            self.reward_func = reward_func

        self.model = model
        self.session_id = session_id
        self.tls_id = tls_id
        self.selected_program_ids = self.get_tls_program_ids(self.initial_data, tls_id)
        self.tls_data = self.get_tls_data(self.initial_data, tls_id)

    def get_selected_action_method(self, action) -> callable:
        if action == self.Action.STEP:
            ret = lambda : None
        elif action == self.Action.NEXT_PHASE:
            ret = lambda : set_traffic_light_phase(self.tls_id, self.session_id,
                                                    make_step=0)  # Set next phase
        else:  # Switch Program
            selected_program_index = action - 2
            if selected_program_index >= len(self.selected_program_ids):
                raise RuntimeError("Illegal action")
            selected_program = self.selected_program_ids[int(selected_program_index)]
            ret = lambda : switch_traffic_light_program(tls_id=self.tls_id, session_id=self.session_id,
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

    @staticmethod
    def default_reward_func(states: dict, cars_that_left: int) -> torch.Tensor:
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

    def step(self, environment_state) -> callable:
        state_tensor, reward, is_ended = self.extract_state_tensor(environment_state)
        if is_ended:
            return None
        self.model.optimize_model()
        selected_action = self.model.select_action(state_tensor, reward)
        return self.get_selected_action_method(selected_action)

