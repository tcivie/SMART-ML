from typing import Callable, Type

import torch
from matplotlib import pyplot as plt

from api_endpoints import start_simulation, reset_simulation, step_simulation, stop_simulation
from experiments.configs import SUMO_SIMULATIONS_BASE_PATH
from experiments.experiments_base import Experiment
from experiments.models.models_base import BaseModel


class ConfigBase:
    def __init__(self,
                 epochs: int,
                 step_size: int,
                 model_class: Type[BaseModel],
                 experiment_class,
                 model_params_func: Callable[[str], BaseModel.Params],
                 simulation_run_path: str = 'bologna/acosta/run.sumocfg',
                 reward_func: Callable[[dict, int], torch.Tensor] = None, *, is_gui=False
                 ):
        self.model = model_class
        self.reward_func = reward_func
        self.simulation_id = start_simulation(str(SUMO_SIMULATIONS_BASE_PATH / simulation_run_path), is_gui=is_gui)
        self.epochs = epochs
        self.step_size = step_size

        reset_simulation(self.simulation_id)
        self.state = step_simulation(self.simulation_id, 0)

        self.agents = [
            experiment_class(self.simulation_id, tls_id,
                             model=self.model(
                                model_params_func(tls_id)
                             ),
                             reward_func=self.reward_func
                             ) for tls_id in self.state['vehicles_in_tls']]

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
            print(f'Epoch\t[{epoch}/{epochs}]\t|\ttotal steps:\t{step}')
            results.append(step)
            step = 0
        stop_simulation(simulation_id)
        # Create the plot
        plt.figure(figsize=(12, 6))
        plt.plot(epochs, results, marker='o', linestyle='-', color='b')
        plt.title('Total Steps per Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Total Steps')
        plt.grid(True)
        plt.xticks(range(0, epochs, 5))
        plt.tight_layout()

        # Display the plot
        plt.show()