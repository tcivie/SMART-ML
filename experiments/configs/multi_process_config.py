import multiprocessing
import os.path
import time
from pathlib import Path

import ray
from ray import train, tune
from typing import Callable, Type

import torch
from torch.utils.tensorboard import SummaryWriter

from experiments.SingleTLS import SumoSingleTLSExperiment, SumoSingleTLSExperimentUncontrolledPhase, \
    SumoSingleTLSExperimentUncontrolledPhaseWithMasterReward, SumoSingleTLSExperimentAllStates
from experiments.configs import reward_functions
from experiments.configs.configs_base import ConfigBase, NoModelConfig
from experiments.configs.reward_functions import penalize_long_wait_times, speed_safety_balance, \
    normalized_general_reward_func
from experiments.experiments_base import Experiment

from experiments.models.DQN import DQN, SplitDQN, DQNWithPhases, LSTMDQNWithPhases
from experiments.models.components.memory import ReplayMemory
from experiments.models.components.networks import SimpleNetwork, SplitNetwork, LSTMNetwork
from experiments.models.models_base import BaseModel
from experiments.models.rewardModel import RewardModel
from sumo_sim.Simulation import LightPhase


def create_and_run_simulation(config,
                              comment: str,
                              step_size,
                              model_class: Type[BaseModel],
                              experiment_class: Type[Experiment],
                              model_params_func: Callable[[object, str, dict], BaseModel.Params],
                              simulation_run_path: str = 'bologna/acosta/run.sumocfg',
                              reward_func: Callable[[dict, int], torch.Tensor] = None,
                              is_gui=True,
                              config_type: Type[ConfigBase] = ConfigBase,
                              ):
    epochs = config["epochs"]

    writer = SummaryWriter(
        comment=f'{model_class.__name__}-{experiment_class.__name__}-{model_params_func.__name__}-{reward_func.__name__}'
    )
    writer.add_text('model_class', str(model_class))
    writer.add_text('experiment_class', str(experiment_class))
    writer.add_text('model_params_func', str(model_params_func))
    writer.add_text('reward_func', str(reward_func))

    writer.add_text('total_epochs', str(epochs))
    writer.add_text('step_size', str(step_size))
    writer.add_text('simulation_run_path', simulation_run_path)
    writer.add_text('is_gui', str(is_gui))
    writer.add_text('comment', comment)

    sim = config_type(epochs,
                      step_size,
                      model_class,
                      experiment_class,
                      writer,
                      lambda state, tls_id: model_params_func(state, tls_id, config),
                      simulation_run_path,
                      reward_func,
                      is_gui=is_gui)
    writer.add_text('config', str(sim))
    sim.run_till_end()
    writer.close()


simulation_run_path = 'bologna/acosta/run.sumocfg'


def empty_reward_func(state, tls_id: str):
    return DQN.Params()


# def simple_network(state, tls_id: str, config):
#     dim_size = len(state['vehicles_in_tls'][tls_id]['lanes'])
#     num_controlled_links = state['num_controlled_links'][tls_id]
#     states = sum(t['total_phases_count'] for t in state['vehicles_in_tls'].values())
#     network_input = 7 * dim_size + states
#     network_output = 2  # Actions ( Do nothing or change phase )
#     try:
#         hidden_layers = [config["layer1_size"], config["layer2_size"], config["layer3_size"]][:config["num_layers"]]
#     except:
#         hidden_layers = [64, 64]
#     policy_net = SimpleNetwork(network_input,
#                                network_output, hidden_layers)
#     target_net = SimpleNetwork(network_input,
#                                network_output, hidden_layers)
#
#     return DQN.Params(
#         observations=network_input,
#         policy_net=policy_net,
#         target_net=target_net,
#
#         optimizer=torch.optim.Adam(policy_net.parameters(), lr=config["lr"]),
#         memory=ReplayMemory(config["memory_size"]),
#
#         EPS_START=config["eps_start"],
#         EPS_END=config["eps_end"],
#         EPS_DECAY=config["eps_decay"],
#         GAMMA=config["gamma"],
#         BATCH_SIZE=config["batch_size"],
#         TARGET_UPDATE=config["target_update"],
#     )

def simple_network(state, tls_id: str, config):
    dim_size = len(state['vehicles_in_tls'][tls_id]['lanes'])
    network_input = 7 * dim_size + state['vehicles_in_tls'][tls_id]['total_phases_count']
    network_output = 2  # Actions ( Do nothing or change phase )
    try:
        hidden_layers = [config["layer1_size"], config["layer2_size"], config["layer3_size"]][:config["num_layers"]]
    except:
        hidden_layers = [64, 64]
    policy_net = SimpleNetwork(network_input,
                               network_output, hidden_layers)
    target_net = SimpleNetwork(network_input,
                               network_output, hidden_layers)

    return DQN.Params(
        observations=network_input,
        policy_net=policy_net,
        target_net=target_net,

        optimizer=torch.optim.Adam(policy_net.parameters(), lr=config["lr"]),
        memory=ReplayMemory(config["memory_size"]),

        EPS_START=config["eps_start"],
        EPS_END=config["eps_end"],
        EPS_DECAY=config["eps_decay"],
        GAMMA=config["gamma"],
        BATCH_SIZE=config["batch_size"],
        TARGET_UPDATE=config["target_update"],
    )

if __name__ == '__main__':
    logs_path = Path(__file__).parent / 'runs'
    param_space = {
        "epochs": tune.randint(10, 20),
        "lr": tune.loguniform(1e-4, 1e-2),
        "batch_size": tune.choice([32, 64, 128]),
        "gamma": tune.uniform(0.5, 0.99),
        "layer1_size": tune.choice([16, 32, 64]),
        "layer2_size": tune.choice([16, 32, 64]),
        "layer3_size": tune.choice([16, 32, 64]),
        "num_layers": tune.choice([1, 2, 3]),
        "memory_size": tune.choice([500, 1000, 2000]),
        "eps_start": tune.uniform(0.5, 1.0),
        "eps_end": tune.uniform(0.01, 0.1),
        "eps_decay": tune.randint(500, 2000),
        "target_update": tune.randint(500, 2000)
    }

    comment = """Simple model, where the TLS schedule is controlled by the model and the schedule (If the model wants it can 
    trigger the schedule change) and if not it would be triggered like in the normal simulation. essentially it 
    can only make the TLS change state earlier"""
    tuner = tune.Tuner(
        tune.with_parameters(create_and_run_simulation,
                             comment=comment,
                             step_size=10,
                             model_class=DQN,
                             experiment_class=SumoSingleTLSExperiment,
                             model_params_func=simple_network,
                             simulation_run_path=simulation_run_path,
                             reward_func=normalized_general_reward_func,
                             is_gui=False,
                             config_type=ConfigBase),
        param_space=param_space,
        tune_config=tune.TuneConfig(
            metric="TotalSteps",
            mode="min",
            num_samples=4,
            trial_dirname_creator=lambda trial: f"{trial.trainable_name}_{trial.trial_id}",
        ),
        run_config=train.RunConfig(
            name="hyperparameter_tuning",
            storage_path=str(logs_path),
        )
    )
    tuner.fit()
    print("Hyperparameter tuning is complete.")
