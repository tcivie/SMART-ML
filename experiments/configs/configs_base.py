import datetime
import time
from typing import List
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import base64
from io import BytesIO
from typing import Callable, Type
import time
from typing import Callable, Type
from pathlib import Path

import torch
from matplotlib import pyplot as plt
import xml.etree.ElementTree as ET
from typing import List, Dict

from torch.utils.tensorboard import SummaryWriter

from api_endpoints import start_simulation, reset_simulation, step_simulation, stop_simulation
from experiments import device
from experiments.configs import SUMO_SIMULATIONS_BASE_PATH
from experiments.experiments_base import Experiment
from experiments.models.models_base import BaseModel

import lxml.etree as lxml_etree


class ConfigBase:
    def __init__(self,
                 epochs: int,
                 step_size: int,
                 model_class: Type[BaseModel],
                 experiment_class,
                 writer: SummaryWriter,
                 model_params_func: Callable[[object, str], BaseModel.Params],
                 simulation_run_path: str = 'bologna/acosta/run.sumocfg',
                 reward_func: Callable[[dict, int], torch.Tensor] = None, *, is_gui=False
                 ):
        self.writer = writer
        self.model = model_class
        self.reward_func = reward_func

        self.simulation_id = start_simulation(str(SUMO_SIMULATIONS_BASE_PATH / simulation_run_path), is_gui=is_gui)
        self.writer.add_text('simulation_id', self.simulation_id)

        self.epochs = epochs
        self.total_steps = 0
        self.step_size = step_size

        reset_simulation(self.simulation_id)
        self.state = step_simulation(self.simulation_id, 0)
        self.log_state_to_tensorboard(self.state)

        self.agents = [
            experiment_class(self.simulation_id,
                             tls_id,
                             model=self.model(
                                 model_params_func(self.state, tls_id)
                             ),
                             reward_func=self.reward_func
                             ) for tls_id in self.state['vehicles_in_tls']]

        # Cannot log multiple models so we would log only the first one
        agent_policy_net = self.agents[0].model.policy_net
        example_input = torch.zeros(1, len(self.state['vehicles_in_tls'][self.agents[0].tls_id]['lanes']) * 7).to(
            device)
        self.writer.add_graph(agent_policy_net, example_input)
        self.log = ConfigLogging(**vars(self))

    def __str__(self):
        return (f"ConfigBase(\n"
                f"  epochs={self.epochs},\n"
                f"  step_size={self.step_size},\n"
                f"  model_class={self.model.__name__},\n"
                f"  simulation_id={self.simulation_id},\n"
                f"  reward_func={'defined' if self.reward_func else 'undefined'},\n"
                f"  agents_count={len(self.agents)},\n"
                f"  agents={str(self.agents)},\n"
                f"  total_steps={self.total_steps},\n"
                f"  state={self.state}\n"
                f")")

    def log_state_to_tensorboard(self, state: dict, step: int = 0):
        self.writer.add_scalar('SimulationMetrics/CarsThatLeft', state.get('cars_that_left', 0), step)
        self.writer.add_scalar('SimulationMetrics/DeltaCarsInTLS', state.get('delta_cars_in_tls', 0), step)

        aggregated_data = {}

        for tls_id, tls in state.get('vehicles_in_tls', {}).items():
            car_metrics = tls.get('longest_waiting_time_car_in_lane', {})

            if tls_id not in aggregated_data:
                aggregated_data[tls_id] = {
                    'total_cars': 0,
                    'average_speed_sum': 0,
                    'max_wait_time_sum': 0,
                    'min_wait_time_sum': 0,
                    'occupancy_sum': 0,
                    'queue_length_sum': 0,
                    'total_cars_sum': 0,
                    'total_co2_emission_sum': 0,
                    'lanes_count': 0
                }

            aggregated_data[tls_id]['total_cars'] += tls.get('total', 0)

            for lane, metrics in car_metrics.items():
                aggregated_data[tls_id]['average_speed_sum'] += metrics.get('average_speed', 0) * 3.6  # Convert to km/h
                aggregated_data[tls_id]['max_wait_time_sum'] += metrics.get('max_wait_time', 0)
                aggregated_data[tls_id]['min_wait_time_sum'] += metrics.get('min_wait_time', 0)
                aggregated_data[tls_id]['occupancy_sum'] += metrics.get('occupancy', 0)
                aggregated_data[tls_id]['queue_length_sum'] += metrics.get('queue_length', 0)
                aggregated_data[tls_id]['total_cars_sum'] += metrics.get('total_cars', 0)
                aggregated_data[tls_id]['total_co2_emission_sum'] += metrics.get('total_co2_emission', 0)
                aggregated_data[tls_id]['lanes_count'] += 1

        for tls_id, metrics in aggregated_data.items():
            lanes_count = metrics['lanes_count']
            self.writer.add_scalar(f'LongestWaitingTime/{tls_id}/TotalCars', metrics['total_cars'], step)
            if lanes_count > 0:
                self.writer.add_scalar(f'LongestWaitingTime/{tls_id}/AverageSpeed',
                                       metrics['average_speed_sum'] / lanes_count, step)
                self.writer.add_scalar(f'LongestWaitingTime/{tls_id}/MaxWaitingTime',
                                       metrics['max_wait_time_sum'] / lanes_count, step)
                self.writer.add_scalar(f'LongestWaitingTime/{tls_id}/MinWaitingTime',
                                       metrics['min_wait_time_sum'] / lanes_count, step)
                self.writer.add_scalar(f'LongestWaitingTime/{tls_id}/Occupancy', metrics['occupancy_sum'] / lanes_count,
                                       step)
                self.writer.add_scalar(f'LongestWaitingTime/{tls_id}/QueueLength',
                                       metrics['queue_length_sum'] / lanes_count, step)
                self.writer.add_scalar(f'LongestWaitingTime/{tls_id}/TotalCarsSum',
                                       metrics['total_cars_sum'] / lanes_count, step)
                self.writer.add_scalar(f'LongestWaitingTime/{tls_id}/TotalCO2Emission',
                                       metrics['total_co2_emission_sum'] / lanes_count, step)

    def run_till_end(self):
        state = self.state
        simulation_id = self.simulation_id
        epochs = self.epochs
        step_size = self.step_size
        results = []
        step = 0
        print(f"Starting simulation with {epochs} epochs and {step_size} step size")
        for epoch in range(epochs):
            ended = False
            while not ended:
                for agent in self.agents:
                    call = agent.step(state)
                    if call:
                        call()
                    else:
                        reset_simulation(simulation_id)
                        state = step_simulation(simulation_id, 0)
                        ended = True
                        break
                if not ended:
                    state = step_simulation(simulation_id, step_size)
                    step += step_size
                    self.total_steps += step_size
                self.log_state_to_tensorboard(state, self.total_steps)

            self.log.print_epoch(epoch, epochs, step)
            self.writer.add_scalar('SimulationMetrics/TotalSteps', step, epoch)
            results.append(step)
            step = 0
        stop_simulation(simulation_id)


class MasterSlaveConfig(ConfigBase):
    def __init__(self,
                 epochs: int,
                 step_size: int,
                 master_model_class: Type[BaseModel],
                 model_class: Type[BaseModel],
                 experiment_class,
                 writer: SummaryWriter,
                 master_model_params_func: Callable[[object, str], BaseModel.Params],
                 model_params_func: Callable[[object, str], BaseModel.Params],
                 simulation_run_path: str = 'bologna/acosta/run.sumocfg',
                 reward_func: Callable[[dict, int], torch.Tensor] = None,
                 is_gui=False
                 ):
        super().__init__(epochs,
                         step_size,
                         model_class,
                         experiment_class,
                         writer,
                         model_params_func,
                         simulation_run_path=simulation_run_path,
                         reward_func=reward_func,
                         is_gui=is_gui)

    # def run_till_end(self):
    #





class ConfigLogging:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.logs = []
        self.unique_id = id(ConfigLogging)
        self.start_time = time.time()

    def get_formatted_time(self):
        return datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')

    def print_epoch(self, epoch: int, epochs: int, total_steps: int):
        print(
            f'{self.get_formatted_time()}\t|[{self.unique_id}]| Epoch: [{epoch}/{epochs}]\t| total steps: {total_steps}')

    def format_time_elapsed(self, elapsed_time):
        hours, rem = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(rem, 60)
        return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"
