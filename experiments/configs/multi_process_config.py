import multiprocessing
import time
from multiprocessing import freeze_support

from typing import Callable, Type

import torch
from torch.utils.tensorboard import SummaryWriter

from experiments.SingleTLS import SumoSingleTLSExperiment, SumoSingleTLSExperimentUncontrolledPhase
from experiments.configs import reward_functions
from experiments.configs.configs_base import ConfigBase

from experiments.models.DQN import DQN, SplitDQN
from experiments.models.components.networks import SimpleNetwork, SplitNetwork
from experiments.models.models_base import BaseModel
from sumo_sim.Simulation import LightPhase


def create_and_run_simulation(epochs: int,
                              step_size: int,
                              model_class: Type[BaseModel],
                              experiment_class,
                              model_params_func: Callable[[object, str], BaseModel.Params],
                              simulation_run_path: str = 'bologna/acosta/run.sumocfg',
                              reward_func: Callable[[dict, int], torch.Tensor] = None, *, is_gui=False):
    writer = SummaryWriter(
        comment=f'{model_class.__name__}_{experiment_class.__name__}_{model_params_func.__name__}_{reward_func.__name__}'
    )
    writer.add_text('model_class', model_class.__name__)
    writer.add_text('experiment_class', experiment_class.__name__)
    writer.add_text('model_params_func', model_params_func.__name__)
    writer.add_text('reward_func', reward_func.__name__)

    writer.add_text('total_epochs', str(epochs))
    writer.add_text('step_size', str(step_size))
    writer.add_text('simulation_run_path', simulation_run_path)
    writer.add_text('is_gui', str(is_gui))

    sim = ConfigBase(epochs,
                     step_size,
                     model_class,
                     experiment_class,
                     writer,
                     model_params_func,
                     simulation_run_path,
                     reward_func, is_gui=is_gui)
    sim.run_till_end()
    writer.close()


simulation_run_path = 'bologna/acosta/run.sumocfg'


# def hidden_2(dim_size: int):
#     policy_net = SimpleNetwork(7 * dim_size,
#                                2, [64, 64])
#     target_net = SimpleNetwork(7 * dim_size,
#                                2, [64, 64])
#     return DQN.Params(
#         observations=7 * dim_size,
#         actions=2,
#         policy_net=policy_net,
#         target_net=target_net,
#         optimizer=torch.optim.Adam(policy_net.parameters(), lr=0.001),
#     )
#
#
# def hidden_3(dim_size: int):
#     policy_net = SimpleNetwork(7 * dim_size,
#                                2, [64, 64, 64])
#     target_net = SimpleNetwork(7 * dim_size,
#                                2, [64, 64, 64])
#     return DQN.Params(
#         observations=7 * dim_size,
#         actions=2,
#         policy_net=policy_net,
#         target_net=target_net,
#         optimizer=torch.optim.Adam(policy_net.parameters(), lr=0.001),
#     )
#
#
# def hidden_3_small(dim_size: int):
#     policy_net = SimpleNetwork(7 * dim_size,
#                                2, [7 * dim_size, 7 * dim_size, 7 * dim_size])
#     target_net = SimpleNetwork(7 * dim_size,
#                                2, [7 * dim_size, 7 * dim_size, 7 * dim_size])
#     return DQN.Params(
#         observations=7 * dim_size,
#         actions=2,
#         policy_net=policy_net,
#         target_net=target_net,
#         optimizer=torch.optim.Adam(policy_net.parameters(), lr=0.001),
#     )
#
#
# def hidden_4(dim_size: int):
#     policy_net = SimpleNetwork(7 * dim_size,
#                                2, [64, 64, 64, 64])
#     target_net = SimpleNetwork(7 * dim_size,
#                                2, [64, 64, 64, 64])
#     return DQN.Params(
#         observations=7 * dim_size,
#         actions=2,
#         policy_net=policy_net,
#         target_net=target_net,
#         optimizer=torch.optim.Adam(policy_net.parameters(), lr=0.001),
#     )

def hidden_2x2(state, tls_id: str):
    dim_size = len(state['vehicles_in_tls'][tls_id]['lanes'])
    num_controlled_links = state['num_controlled_links'][tls_id]
    policy_net = SplitNetwork(7 * dim_size, num_controlled_links * len(LightPhase), [[64, 32, 64], [32, 16, 32]], 1)
    target_net = SplitNetwork(7 * dim_size, num_controlled_links * len(LightPhase), [[64, 32, 64], [32, 16, 32]], 1)
    return SplitDQN.Params(
        observations=7 * dim_size,
        policy_net=policy_net,
        target_net=target_net,
        optimizer=torch.optim.Adam(policy_net.parameters(), lr=0.001),
        num_of_controlled_links=num_controlled_links
    )


if __name__ == '__main__':

    # List of arguments to pass to the function
    args1 = (
        20,
        10,
        SplitDQN,
        SumoSingleTLSExperimentUncontrolledPhase,
        hidden_2x2,
        simulation_run_path,
        reward_functions.even_traffic_distribution)
    # args2 = (
    #     50, 30, DQN, SumoSingleTLSExperiment,
    #     hidden_3,
    #     simulation_run_path,
    #     reward_functions.environmental_impact)
    # args3 = (
    #     50, 30, DQN, SumoSingleTLSExperiment,
    #     hidden_3_small,
    #     simulation_run_path,
    #     reward_functions.environmental_impact)
    # args4 = (
    #     50, 30, DQN, SumoSingleTLSExperiment,
    #     hidden_4,
    #     simulation_run_path,
    #     reward_functions.environmental_impact)

    # arguments = [args1, args2, args3, args4]
    arguments = [args1]
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
