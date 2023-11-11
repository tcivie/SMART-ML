import math
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
    "-c", r"C:\Users\omera\Desktop\College\Year 4\Final Project\Cloned\scenarios\bologna\acosta\run.sumocfg"
]


class Lane:
    def __init__(self, lid):
        self._id = lid
        self._shape = traci.lane.getShape(lid)

    def get_min_distance_points(self, other):
        min_distance = math.inf
        point1 = self._shape[0]
        point2 = other._shape[0]

        for my_point in self._shape:
            for other_point in other._shape:
                distance = math.dist(my_point, other_point)
                if distance < min_distance:
                    min_distance = distance
                    point1 = my_point
                    point2 = other_point

        return min_distance, point1, point2

    def __str__(self):
        return 'Lane: {}'.format(self._id)

    def __repr__(self):
        return self.__str__()


class TrafficLight:
    def __init__(self, tid):
        self._id = tid
        self._lanes = [Lane(i) for i in traci.trafficlight.getControlledLanes(tid)]
        self._links = traci.trafficlight.getControlledLinks(tid)

    def __str__(self):
        return 'TrafficLight: {}'.format(self._id)

    def __repr__(self):
        return self.__str__()


# Start TraCI with the SUMO command
traci.start(sumoCmd)
traffic_light_ids = traci.trafficlight.getIDList()
light1 = TrafficLight(traffic_light_ids[0])

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
