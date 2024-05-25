"""
n_observations = len(selected_tls['lanes'] * 7)
programs_count = len(selected_program_ids)
n_actions = 2 + programs_count # Step, Change Phase, Change Program

print(f'Observations: {n_observations} | Actions: {n_actions}')
"""
from matplotlib import pyplot as plt

from api_endpoints import start_simulation, get_initial_data, reset_simulation, step_simulation, stop_simulation
from experiments.SingleTLS import SumoSingleTLSExperiment
from experiments.configs import SUMO_SIMULATIONS_BASE_PATH
from experiments.configs.reward_functions import penalize_long_wait_times
from experiments.models.DQN import DQN
from experiments.models.components.networks import SimpleNetwork

# Start the simulation
simulation_id = start_simulation(str(SUMO_SIMULATIONS_BASE_PATH / 'bologna/acosta/run.sumocfg'), is_gui=False)
reset_simulation(simulation_id)

# Get the initial data
_ = get_initial_data(simulation_id)
state = step_simulation(simulation_id, 0)

agents = [SumoSingleTLSExperiment(simulation_id, tls_id,
                                  model=DQN(
                                      DQN.Params(
                                          observations=7 * len(state['vehicles_in_tls'][tls_id]['lanes']),
                                          actions=4,
                                          # Step (1) + Next Phase (1) + Switch Program (Number of programs) (2) = 4
                                          policy_net=SimpleNetwork(7 * len(state['vehicles_in_tls'][tls_id]['lanes']),
                                                                   3, [64, 128, 64]),
                                          target_net=SimpleNetwork(7 * len(state['vehicles_in_tls'][tls_id]['lanes']),
                                                                   3, [64, 128, 64])
                                      )
                                  ),
                                  reward_func=penalize_long_wait_times)
          for tls_id in state['vehicles_in_tls']]

ended = False
step = 0
step_size = 30
epochs = 20

results = []

print(f"Starting simulation with {epochs} epochs and {step_size} step size")
for epoch in range(epochs):
    ended = False
    while not ended:
        for agent in agents:
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