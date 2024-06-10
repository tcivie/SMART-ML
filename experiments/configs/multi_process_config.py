import multiprocessing
import subprocess
import sys
import time
from multiprocessing import freeze_support

from typing import Callable, Type

import torch
from torch.utils.tensorboard import SummaryWriter

from experiments.SingleTLS import SumoSingleTLSExperiment, SumoSingleTLSExperimentUncontrolledPhase, \
    SumoSingleTLSExperimentUncontrolledPhaseWithMasterReward
from experiments.configs import reward_functions
from experiments.configs.configs_base import ConfigBase

from experiments.models.DQN import DQN, SplitDQN, DQNWithPhases, LSTMDQNWithPhases
from experiments.models.components.memory import ReplayMemory
from experiments.models.components.networks import SimpleNetwork, SplitNetwork, LSTMNetwork
from experiments.models.models_base import BaseModel
from experiments.models.rewardModel import RewardModel
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
    writer.add_text('model_class', str(model_class))
    writer.add_text('experiment_class', str(experiment_class))
    writer.add_text('model_params_func', str(model_params_func))
    writer.add_text('reward_func', str(reward_func))

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
    writer.add_text('config', str(sim))
    sim.run_till_end()
    writer.close()


simulation_run_path = 'bologna/acosta/run.sumocfg'


def hidden_2(state, tls_id: str):
    dim_size = len(state['vehicles_in_tls'][tls_id]['lanes'])
    num_controlled_links = state['num_controlled_links'][tls_id]
    policy_net = SimpleNetwork(7 * dim_size,
                               num_controlled_links * len(LightPhase), [64, 64])
    target_net = SimpleNetwork(7 * dim_size,
                               num_controlled_links * len(LightPhase), [64, 64])
    return DQNWithPhases.Params(
        observations=7 * dim_size,
        policy_net=policy_net,
        target_net=target_net,
        optimizer=torch.optim.Adam(policy_net.parameters(), lr=0.001),
        num_of_controlled_links=num_controlled_links,

        memory=ReplayMemory(100_000),
        EPS_START=0.9,
        EPS_END=0.05,
        EPS_DECAY=1_000,
        GAMMA=0.9,
        BATCH_SIZE=32,
        TARGET_UPDATE=1_000
    )


#
#
def hidden_3(state, tls_id: str):
    dim_size = len(state['vehicles_in_tls'][tls_id]['lanes'])
    num_controlled_links = state['num_controlled_links'][tls_id]
    policy_net = SimpleNetwork(7 * dim_size,
                               num_controlled_links * len(LightPhase), [64, 64, 64])
    target_net = SimpleNetwork(7 * dim_size,
                               num_controlled_links * len(LightPhase), [64, 64, 64])
    return DQNWithPhases.Params(
        observations=7 * dim_size,
        policy_net=policy_net,
        target_net=target_net,
        optimizer=torch.optim.Adam(policy_net.parameters(), lr=0.001),
        num_of_controlled_links=num_controlled_links,

        memory=ReplayMemory(100_000),
        EPS_START=0.9,
        EPS_END=0.05,
        EPS_DECAY=1_000,
        GAMMA=0.9,
        BATCH_SIZE=32,
        TARGET_UPDATE=1_000
    )


#
#
def hidden_3_small(state, tls_id: str):
    dim_size = len(state['vehicles_in_tls'][tls_id]['lanes'])
    num_controlled_links = state['num_controlled_links'][tls_id]
    policy_net = SimpleNetwork(7 * dim_size,
                               num_controlled_links * len(LightPhase), [7 * dim_size, dim_size, dim_size // 2])
    target_net = SimpleNetwork(7 * dim_size,
                               num_controlled_links * len(LightPhase), [7 * dim_size, dim_size, dim_size // 2])
    return DQNWithPhases.Params(
        observations=7 * dim_size,
        policy_net=policy_net,
        target_net=target_net,
        optimizer=torch.optim.Adam(policy_net.parameters(), lr=0.001),
        num_of_controlled_links=num_controlled_links,

        memory=ReplayMemory(100_000),
        EPS_START=0.9,
        EPS_END=0.05,
        EPS_DECAY=1_000,
        GAMMA=0.9,
        BATCH_SIZE=32,
        TARGET_UPDATE=1_000
    )


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


# @dataclass
#     class Params(BaseModel.Params):
#         observations: int = 7
#         policy_net: nn.Module = SimpleNetwork(7, 3, [64, 64])
#         target_net: nn.Module = SimpleNetwork(7, 3, [64, 64])
#         optimizer: torch.optim.Optimizer = torch.optim.Adam
#         memory: ReplayMemory = ReplayMemory(10000)
#
#         EPS_START: float = 0.9
#         EPS_END: float = 0.05
#         EPS_DECAY: float = 200
#
#         GAMMA: float = 0.999
#         BATCH_SIZE: int = 128
def hidden_2x2(state, tls_id: str):
    dim_size = len(state['vehicles_in_tls'][tls_id]['lanes'])
    num_controlled_links = state['num_controlled_links'][tls_id]
    policy_net = SplitNetwork(7 * dim_size, num_controlled_links * len(LightPhase),
                              [[64, 32, 16], [32, 16]], 1)
    target_net = SplitNetwork(7 * dim_size, num_controlled_links * len(LightPhase),
                              [[64, 32, 16], [32, 16]], 1)
    return SplitDQN.Params(
        observations=7 * dim_size,
        policy_net=policy_net,
        target_net=target_net,
        optimizer=torch.optim.Adam(policy_net.parameters(), lr=0.1),
        num_of_controlled_links=num_controlled_links,

        memory=ReplayMemory(100_000),
        EPS_START=0.9,
        EPS_END=0.05,
        EPS_DECAY=1_000,
        GAMMA=0.9,
        BATCH_SIZE=32,
        TARGET_UPDATE=1_000
    )

def lstm_net_tiny(state, tls_id: str):
    dim_size = len(state['vehicles_in_tls'][tls_id]['lanes'])
    num_controlled_links = state['num_controlled_links'][tls_id]
    policy_net = LSTMNetwork(7 * dim_size, 32, 1, num_controlled_links * len(LightPhase))
    target_net = LSTMNetwork(7 * dim_size, 32, 1, num_controlled_links * len(LightPhase))
    return DQNWithPhases.Params(
        observations=7 * dim_size,
        policy_net=policy_net,
        target_net=target_net,
        optimizer=torch.optim.Adam(policy_net.parameters(), lr=0.001),
        num_of_controlled_links=num_controlled_links,

        memory=ReplayMemory(100_000),
        EPS_START=0.9,
        EPS_END=0.05,
        EPS_DECAY=1_000,
        GAMMA=0.9,
        BATCH_SIZE=32,
        TARGET_UPDATE=1_000
    )

def lstm_net_small(state, tls_id: str):
    dim_size = len(state['vehicles_in_tls'][tls_id]['lanes'])
    num_controlled_links = state['num_controlled_links'][tls_id]
    policy_net = LSTMNetwork(7 * dim_size, 64, 2, num_controlled_links * len(LightPhase))
    target_net = LSTMNetwork(7 * dim_size, 64, 2, num_controlled_links * len(LightPhase))
    return DQNWithPhases.Params(
        observations=7 * dim_size,
        policy_net=policy_net,
        target_net=target_net,
        optimizer=torch.optim.Adam(policy_net.parameters(), lr=0.001),
        num_of_controlled_links=num_controlled_links,

        memory=ReplayMemory(200_000),
        EPS_START=0.8,
        EPS_END=0.05,
        EPS_DECAY=1_000,
        GAMMA=0.95,
        BATCH_SIZE=64,
        TARGET_UPDATE=500
    )

def lstm_net_medium(state, tls_id: str):
    dim_size = len(state['vehicles_in_tls'][tls_id]['lanes'])
    num_controlled_links = state['num_controlled_links'][tls_id]
    policy_net = LSTMNetwork(7 * dim_size, 128, 3, num_controlled_links * len(LightPhase))
    target_net = LSTMNetwork(7 * dim_size, 128, 3, num_controlled_links * len(LightPhase))
    return DQNWithPhases.Params(
        observations=7 * dim_size,
        policy_net=policy_net,
        target_net=target_net,
        optimizer=torch.optim.Adam(policy_net.parameters(), lr=0.001),
        num_of_controlled_links=num_controlled_links,

        memory=ReplayMemory(500_000),
        EPS_START=0.7,
        EPS_END=0.05,
        EPS_DECAY=500,
        GAMMA=0.99,
        BATCH_SIZE=128,
        TARGET_UPDATE=500
    )

def lstm_net_large(state, tls_id: str):
    dim_size = len(state['vehicles_in_tls'][tls_id]['lanes'])
    num_controlled_links = state['num_controlled_links'][tls_id]
    policy_net = LSTMNetwork(7 * dim_size, 256, 4, num_controlled_links * len(LightPhase))
    target_net = LSTMNetwork(7 * dim_size, 256, 4, num_controlled_links * len(LightPhase))
    return DQNWithPhases.Params(
        observations=7 * dim_size,
        policy_net=policy_net,
        target_net=target_net,
        optimizer=torch.optim.Adam(policy_net.parameters(), lr=0.0005),
        num_of_controlled_links=num_controlled_links,

        memory=ReplayMemory(1_000_000),
        EPS_START=0.6,
        EPS_END=0.05,
        EPS_DECAY=500,
        GAMMA=0.99,
        BATCH_SIZE=256,
        TARGET_UPDATE=200
    )


if __name__ == '__main__':
    if 'darwin' in sys.platform:
        print('Running \'caffeinate\' on MacOSX to prevent the system from sleeping')
    subprocess.Popen('caffeinate')
    # List of arguments to pass to the function
    args1 = (
        20,
        10,
        LSTMDQNWithPhases,
        SumoSingleTLSExperimentUncontrolledPhaseWithMasterReward,
        lstm_net_tiny,
        simulation_run_path,
        RewardModel
    )
    args2 = (
        20,
        10,
        LSTMDQNWithPhases,
        SumoSingleTLSExperimentUncontrolledPhaseWithMasterReward,
        lstm_net_small,
        simulation_run_path,
        RewardModel
    )
    args3 = (
        20,
        10,
        LSTMDQNWithPhases,
        SumoSingleTLSExperimentUncontrolledPhaseWithMasterReward,
        lstm_net_medium,
        simulation_run_path,
        RewardModel
    )
    args4 = (
        20,
        10,
        LSTMDQNWithPhases,
        SumoSingleTLSExperimentUncontrolledPhaseWithMasterReward,
        lstm_net_large,
        simulation_run_path,
        RewardModel
    )

    arguments = [args1, args2, args3, args4]
    # arguments = [args1]
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
