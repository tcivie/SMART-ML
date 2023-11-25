import os
import sys

import traci

# Check for the SUMO_HOME environment variable
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

# This is the command to run SUMO in headless mode
sumoCmd = [
    "sumo-gui",
    "-c", r"C:\Users\omera\Desktop\College\Year 4\Final Project\Cloned\dockerImage\scenarios\bologna\acosta\run.sumocfg"
]

# Start TraCI with the SUMO command
traci.start(sumoCmd)
traffic_light_ids = traci.trafficlight.getIDList()

# Iterate through traffic lights and get legal phases
for tl_id in traffic_light_ids:
    # Get the controlled connections for the traffic light
    controlled_connections = traci.trafficlight.getControlledLinks(tl_id)

    # Extract the phases from the controlled connections
    legal_phases = set()
    for conn in controlled_connections:
        for connection in conn:
            # Each connection is a tuple (incoming lane, outgoing lane, direction)
            _, _, direction = connection
        legal_phases.add(direction)

    # Print the legal phases for the traffic light
    print(f"Traffic Light {tl_id} has legal phases: {legal_phases}")
# Simulation loop
step = 0
while traci.simulation.getMinExpectedNumber() > 0:  # Run until all vehicles have left the network
    traci.simulationStep()
    step += 1

# Get the list of all traffic light IDs
traffic_lights = traci.trafficlight.getIDList()
print("Traffic Lights in the simulation:", traffic_lights)

# End the TraCI connection
traci.close()

# Optional: Print how many steps the simulation took
print(f"Simulation finished in {step} steps")
