import multiprocessing
import time
from multiprocessing import freeze_support

from typing import Callable, Type

import torch

from experiments.SingleTLS import SumoSingleTLSExperiment
from experiments.configs import reward_functions
from experiments.configs.configs_base import ConfigBase

from experiments.models.DQN import DQN
from experiments.models.components.networks import SimpleNetwork
from experiments.models.models_base import BaseModel


def create_and_run_simulation(epochs: int,
                              step_size: int,
                              model_class: Type[BaseModel],
                              experiment_class,
                              model_params_func: Callable[[int], BaseModel.Params],
                              simulation_run_path: str = 'bologna/acosta/run.sumocfg',
                              reward_func: Callable[[dict, int], torch.Tensor] = None, *, is_gui=False):
    sim = ConfigBase(epochs,
                     step_size,
                     model_class,
                     experiment_class,
                     model_params_func,
                     simulation_run_path,
                     reward_func, is_gui=is_gui)
    sim.run_till_end()


simulation_run_path = 'bologna/acosta/run.sumocfg'


#len(state['vehicles_in_tls'][tls_id]['lanes'])
def params_func(dim_size: int):
    return DQN.Params(
        observations=7 * dim_size,
        actions=4,
        # Step (1) + Next Phase (1) + Switch Program (Number of programs) (2) = 4
        policy_net=SimpleNetwork(7 * dim_size,
                                 3, [64, 128, 64]),
        target_net=SimpleNetwork(7 * dim_size,
                                 3, [64, 128, 64])
    )


if __name__ == '__main__':

    # List of arguments to pass to the function
    args1 = (
        2, 30, DQN, SumoSingleTLSExperiment,
        lambda dim_size: DQN.Params(
        observations=7 * dim_size,
        actions=4,
        # Step (1) + Next Phase (1) + Switch Program (Number of programs) (2) = 4
        policy_net=SimpleNetwork(7 * dim_size,
                                 3, [64, 64]),
        target_net=SimpleNetwork(7 * dim_size,
                                 3, [64, 64])
    )
        , simulation_run_path,
        reward_functions.environmental_impact)
    args2 = (
        50, 30, DQN, SumoSingleTLSExperiment,
        lambda dim_size: DQN.Params(
            observations=7 * dim_size,
            actions=4,
            # Step (1) + Next Phase (1) + Switch Program (Number of programs) (2) = 4
            policy_net=SimpleNetwork(7 * dim_size,
                                     3, [64, 64, 64]),
            target_net=SimpleNetwork(7 * dim_size,
                                     3, [64, 64, 64])
        )
        , simulation_run_path, reward_functions.environmental_impact)
    args3 = (
        50, 30, DQN, SumoSingleTLSExperiment,
        lambda dim_size: DQN.Params(
            observations=7 * dim_size,
            actions=4,
            # Step (1) + Next Phase (1) + Switch Program (Number of programs) (2) = 4
            policy_net=SimpleNetwork(7 * dim_size,
                                     3, [64, 64, 64, 64]),
            target_net=SimpleNetwork(7 * dim_size,
                                     3, [64, 64, 64, 64])
        )
        , simulation_run_path, reward_functions.environmental_impact)
    args4 = (
        50, 30, DQN, SumoSingleTLSExperiment,
        lambda dim_size: DQN.Params(
            observations=7 * dim_size,
            actions=4,
            # Step (1) + Next Phase (1) + Switch Program (Number of programs) (2) = 4
            policy_net=SimpleNetwork(7 * dim_size,
                                     3, [7 * dim_size, 128, 64]),
            target_net=SimpleNetwork(7 * dim_size,
                                     3, [7 * dim_size, 128, 64])
        )
        , simulation_run_path,
        reward_functions.environmental_impact)

    arguments = [args1, args2, args3, args4]
    create_and_run_simulation(*args1)
    exit(1)
    # Create a list to hold the processes
    processes = []

    # Start a process for each argument
    for arg in arguments:
        p = multiprocessing.Process(target=create_and_run_simulation, args=arg)
        processes.append(p)
        p.start()
        time.sleep(2)

    # Wait for all processes to finish
    for p in processes:
        p.join()

    print("All subprocesses are done.")
