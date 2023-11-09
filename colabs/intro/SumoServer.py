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
    "-c", "/Users/glebtcivie/Documents/Projects/PycharmProjects/SMART-ML/scenarios/bologna/acosta/run.sumocfg"
]

# Start TraCI with the SUMO command
traci.start(sumoCmd)

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