import os
import traci
from pathlib import Path

# Retrieve the SUMO configuration file name from the environment variable
sumo_conf_file = os.getenv('SUMO_CONF_FILE', 'run.sumocfg')

# Construct the full path to the configuration file in /sumo-data directory
config_path = Path('/sumo-data') / sumo_conf_file

# The command to start SUMO, using the full path to the configuration file
sumoCmd = ["sumo", "-c", str(config_path)]

# Start SUMO with TraCI
print("Starting SUMO with TraCI...")
traci.start(sumoCmd)
print("SUMO started.")

# Run the simulation for a certain number of steps
for step in range(1000):  # Adjust the number of steps as needed
    traci.simulationStep()

    # Display the current simulation time
    print(f"Simulation time: {traci.simulation.getTime()}")

    # Print the IDs of all vehicles in the simulation
    vehicles = traci.vehicle.getIDList()
    print(f"Vehicles on the road: {vehicles}")

    # Get and print the speed of each vehicle
    for veh_id in vehicles:
        speed = traci.vehicle.getSpeed(veh_id)
        print(f"Vehicle {veh_id} speed: {speed} m/s")

    # Example: Manipulating traffic lights (if applicable)
    # Get the IDs of all traffic lights
    traffic_lights = traci.trafficlight.getIDList()
    for tl_id in traffic_lights:
        # You can change the state of traffic lights here
        # traci.trafficlight.setRedYellowGreenState(tl_id, "GreenState")
        pass

# Close the TraCI connection
traci.close()
print("TraCI connection closed.")
