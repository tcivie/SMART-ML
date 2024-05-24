"""
n_observations = len(selected_tls['lanes'] * 7)
programs_count = len(selected_program_ids)
n_actions = 2 + programs_count # Step, Change Phase, Change Program

print(f'Observations: {n_observations} | Actions: {n_actions}')
"""
from api_endpoints import start_simulation, get_initial_data, reset_simulation, step_simulation
from experiments.SingleTLS import SumoSingleTLSExperiment
from experiments.configs import SUMO_SIMULATIONS_BASE_PATH
from experiments.models.DQN import DQN, Params
from experiments.models.components.networks import SimpleNetwork

# Start the simulation
simulation_id = start_simulation(str(SUMO_SIMULATIONS_BASE_PATH / 'bologna/acosta/run.sumocfg'), is_gui=True)
reset_simulation(simulation_id)

# Get the initial data
_ = get_initial_data(simulation_id)
state = step_simulation(simulation_id, 0)

agents = [SumoSingleTLSExperiment(simulation_id, tls_id,
                                  DQN(
                                      Params(
                                          observations=7 * len(state['vehicles_in_tls'][tls_id]['lanes']),
                                          actions=3,
                                          policy_net=SimpleNetwork(7 * len(state['vehicles_in_tls'][tls_id]['lanes']),
                                                                   3, [64, 64]),
                                          target_net=SimpleNetwork(7 * len(state['vehicles_in_tls'][tls_id]['lanes']),
                                                                   3, [64, 64])
                                      )
                                  ))
          for tls_id in state['vehicles_in_tls']]

while True:
    for agent in agents:
        is_ended = False
        call = agent.step(state)
        if call:
            call()
        else:
            break
    state = step_simulation(simulation_id, 100)
