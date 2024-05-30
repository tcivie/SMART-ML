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

    def log_state_to_tensorboard(self, state: dict, step: int = 0):

        self.writer.add_scalar('SimulationMetrics/CarsThatLeft', state.get('cars_that_left', 0), step)
        self.writer.add_scalar('SimulationMetrics/DeltaCarsInTLS', state.get('delta_cars_in_tls', 0), step)

        for tls_id in state.get('vehicles_in_tls', {}):
            tls = state['vehicles_in_tls'][tls_id]
            car_metrics = tls.get('longest_waiting_time_car_in_lane', {})

            self.writer.add_scalar(f'LongestWaitingTime/{tls_id}/TotalCars', tls.get('total', 0), step)

            for lane in car_metrics:
                self.writer.add_scalar(f'LongestWaitingTime/{tls_id}/{lane}/AverageSpeed',
                                       car_metrics[lane].get('average_speed', 0), step)
                self.writer.add_scalar(f'LongestWaitingTime/{tls_id}/{lane}/MaxWaitingTime',
                                       car_metrics[lane].get('max_wait_time', 0), step)
                self.writer.add_scalar(f'LongestWaitingTime/{tls_id}/{lane}/MinWaitingTime',
                                       car_metrics[lane].get('min_wait_time', 0), step)
                self.writer.add_scalar(f'LongestWaitingTime/{tls_id}/{lane}/Occupancy',
                                       car_metrics[lane].get('occupancy', 0), step)
                self.writer.add_scalar(f'LongestWaitingTime/{tls_id}/{lane}/QueueLength',
                                       car_metrics[lane].get('queue_length', 0), step)
                self.writer.add_scalar(f'LongestWaitingTime/{tls_id}/{lane}/TotalCars',
                                       car_metrics[lane].get('total_cars', 0), step)
                self.writer.add_scalar(f'LongestWaitingTime/{tls_id}/{lane}/TotalCO2Emission',
                                       car_metrics[lane].get('total_co2_emission', 0), step)

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
            # self.log.log_epoch(epoch, step)
            results.append(step)
            step = 0
        stop_simulation(simulation_id)
        # self.log.plot_results(results, 'Epochs', 'Total Steps', 'Total Steps per Epoch')
        # self.log.summarize_run()
        # self.log.convert_to_html()


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

