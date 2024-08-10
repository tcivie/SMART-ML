from abc import ABC
from enum import Enum
from typing import Callable, Iterable, Type

import torch
from overrides import overrides

from api_endpoints import get_initial_data, set_traffic_light_next_phase, switch_traffic_light_program, \
    set_traffic_light_phase
from experiments import device
from experiments.experiments_base import Experiment
from experiments.models.models_base import BaseModel
from experiments.models.rewardModel import RewardModel
from sumo_sim.Simulation import LightPhase, lights_definitions


class SumoSingleTLSExperiment(Experiment):
    class Action(Enum):
        STEP = 0
        NEXT_PHASE = 1
        # SWITCH_PROGRAM = 2

    def __init__(self, session_id: str, tls_id: str, model: BaseModel,
                 reward_func=None):
        super().__init__(get_initial_data(session_id))
        self.tls_data = self.get_tls_data(self.initial_data, tls_id)
        if reward_func is None:
            self.reward_func = self.default_reward_func
        else:
            if isinstance(reward_func, type):
                self.reward_func = reward_func(self.tls_data)
            else:
                self.reward_func = reward_func

        self.model = model
        self.session_id = session_id
        self.tls_id = tls_id
        self.selected_program_ids = self.get_tls_program_ids(self.initial_data, tls_id)

        self.best_steps_count = float('inf')
        self.current_step_count = 0

        self.current_phase_index = 0
        self.number_of_phases = len(list(
            filter(lambda program: program['program_id'] == self.selected_program_ids[1], self.tls_data['programs']))[
                                        0][
                                        'phases'])  # TODO: Find way to pull current active program and use it instead of index 0 or 1

    def get_selected_action_method(self, action) -> callable:
        if action == self.Action.STEP.value:
            ret = lambda: None
        else:
            ret = lambda: set_traffic_light_next_phase(self.tls_id, self.session_id,
                                                       make_step=0)  # Set next phase
        return ret

    def extract_state_tensor(self, response):
        self.current_step_count += 1
        is_ended = response['is_ended']
        metrics = response['vehicles_in_tls'][self.tls_id]['longest_waiting_time_car_in_lane']
        delta_cars = -response['delta_cars_in_tls']
        extracted_data = []
        for lane in metrics:
            values = list(metrics[lane].values())
            if values:
                extracted_data.extend([float(x) for x in metrics[lane].values()])
            else:
                extracted_data.extend([0. for _ in range(7)])

        # Create tensor for the current phase index
        phase_tensor = torch.zeros(self.number_of_phases, dtype=torch.float32, device=device)
        phase_tensor[response['vehicles_in_tls'][self.tls_id]['current_phase_index']] = 1.0

        # Concatenate phase tensor with state tensor
        state = torch.tensor(extracted_data, dtype=torch.float32, device=device)
        state = torch.cat((state, phase_tensor))

        reward = self.reward_func(metrics, delta_cars)
        if is_ended:
            if self.current_step_count < self.best_steps_count:
                self.best_steps_count = self.current_step_count
                reward += 10  # Bonus for improving the best time
            self.current_step_count = 0  # Reset start time for the next run
        return state, reward, is_ended

    @staticmethod
    def default_reward_func(states: dict, cars_that_left: int) -> torch.Tensor:
        # cars_that_left is basically the cars delta in the tls between the previous step and the current one
        reward = cars_that_left
        for lane in states.values():
            if not lane:
                reward += 10
                continue
            if lane.get('max_wait_time', 0.) > 0:
                # queue_length_percentage = lane['queue_length'] / (lane['total_cars'] / lane['occupancy'])
                reward -= (lane['queue_length'] + lane['max_wait_time'] * 0.5)
            else:
                reward += lane['average_speed']
        return torch.tensor(reward, dtype=torch.float32, device=device)

    def step(self, environment_state) -> callable:
        state_tensor, reward, is_ended = self.extract_state_tensor(environment_state)
        selected_action = self.model.select_action(state_tensor, reward)
        loss = self.model.optimize_model()
        if is_ended:
            return None
        return self.get_selected_action_method(selected_action)


class SumoSingleTLSExperimentUncontrolledPhase(SumoSingleTLSExperiment):
    class Action(Enum):
        STEP = 0
        CHANGE_PHASE = 1

    """
    This class is used to simulate a SumoSingleTLSExperiment where the phase is not controlled by the user. Meaning
    that the phases can change to any state from any state. For example, if in :class:`SumoSingleTLSExperiment` the
    phase is 0, the next phase can be 1, 2, 3 but not 4, 5, 6 due to checks.

    In this class, the next phase that would be returned from a single tls would be list of all lights x phases (
    :class:`LightPhase`) For example if we have simulation single tls which has 4 lights, the number of phases that
    are expected to be returned from the model would be (4*8) = 32. single list with values for each light:[(0-7),
    (0-7), (0-7), (0-7)]

    FYI this class handles only one tls.
    """

    def __init__(self, session_id: str, tls_id: str, model: BaseModel,
                 reward_func: Callable[[dict, int], torch.Tensor] = None):
        super().__init__(session_id, tls_id, model, reward_func)

    @overrides
    def get_selected_action_method(self, action) -> callable:
        if isinstance(action, (int, float)) and action == self.Action.STEP.value:
            ret = lambda: None
        elif isinstance(action, list):
            ret = lambda: set_traffic_light_phase(self.tls_id, self.session_id, action)
        else:
            raise RuntimeError("Illegal action")
        return ret


class SumoSingleTLSExperimentControlledTotalPhases(SumoSingleTLSExperimentUncontrolledPhase):
    def __init__(self, session_id: str, tls_id: str, model: BaseModel,
                 reward_func: Callable[[dict, int], torch.Tensor] = None):
        super().__init__(session_id, tls_id, model, reward_func)
        states = set()
        for program in self.tls_data['programs']:
            for phase in program['phases']:
                states.add(tuple(LightPhase(lights_definitions[phase_id]) for phase_id in phase['state']))
        self.states: list = list(states)

    @overrides
    def get_selected_action_method(self, action) -> callable:
        assert isinstance(action, int), "Invalid Action, should have been int"
        return lambda: set_traffic_light_phase(self.tls_id, self.session_id, self.states[action])


class SumoSingleTLSExperimentUncontrolledPhaseWithMasterReward(SumoSingleTLSExperimentUncontrolledPhase):

    def __init__(self, session_id: str, tls_id: str, model: BaseModel,
                 reward_func: Type[RewardModel] = None):
        super().__init__(session_id, tls_id, model, reward_func)
        if reward_func is not None:
            self.reward_func = reward_func(self.tls_data)


class SumoSingleTLSExperimentAllStates(SumoSingleTLSExperiment):
    def __init__(self, session_id: str, tls_id: str, model: BaseModel,
                 reward_func=None):
        super().__init__(session_id, tls_id, model, reward_func)

    def extract_state_tensor(self, response):
        self.current_step_count += 1
        is_ended = response['is_ended']
        metrics = response['vehicles_in_tls'][self.tls_id]['longest_waiting_time_car_in_lane']
        delta_cars = response['delta_cars_in_tls']
        extracted_data = []
        for lane in metrics:
            values = list(metrics[lane].values())
            if values:
                extracted_data.extend([float(x) for x in metrics[lane].values()])
            else:
                extracted_data.extend([0. for _ in range(7)])

        # Concatenate phase tensor with state tensor
        state = torch.tensor(extracted_data, dtype=torch.float32, device=device)
        state = torch.cat((state,
                           torch.tensor(self.__get_other_tls_state(response), dtype=torch.float32, device=device)))

        reward = self.reward_func(metrics, delta_cars)
        if is_ended:
            if self.current_step_count < self.best_steps_count:
                self.best_steps_count = self.current_step_count
                reward += 10  # Bonus for improving the best time
            self.current_step_count = 0  # Reset start time for the next run
        return state, reward, is_ended

    @staticmethod
    def __get_other_tls_state(response):
        ret = []
        data = response['vehicles_in_tls']
        for tls_id in data:
            tls = data[tls_id]
            add = [0. for _ in range(tls['total_phases_count'])]
            add[data[tls_id]['current_phase_index']] = 1.
            ret.extend(add)
        return ret
