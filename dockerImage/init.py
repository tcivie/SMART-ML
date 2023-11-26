import os
import socket
import uuid
from typing import Union, Any

import sumolib.net
import traci
from flask import Flask, request, jsonify, Response
from functools import lru_cache

app = Flask(__name__)
app.debug = False

# Dictionary to keep track of simulations
simulations = {}  # {session_id: {port: int, conn: traci.connection.Connection, simulation_data: list[dict[str, Any]]}}


def get_connection(session_id) -> traci.connection.Connection:
    global simulations
    return simulations.get(session_id, {}).get('conn')


def start_sumo(config_path: str, port: int, session_id: str) -> Union[traci.connection.Connection, None]:
    print(f"Starting SUMO with config {config_path} on port {port}")
    sumo_args = ['sumo-gui', '-c', config_path]
    label = session_id
    traci.start(sumo_args, port=port, label=label)
    try:
        return traci.getConnection(label)
    except:
        return None


def find_available_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('', 0))
    port = s.getsockname()[1]
    s.close()
    return port


@app.route('/start', methods=['POST'])
def start_simulation():
    """
    Start the simulation
    :return: JSON response
    """
    global simulations
    request_data = request.get_json()
    config_path = request_data.get('config_path')
    if not config_path:
        return "No config path provided", 400

    session_id = str(uuid.uuid4())
    port = find_available_port()
    conn = start_sumo(config_path, port, session_id)
    if not conn:
        return jsonify({
            'success': False,
        })

    simulation_data = get_traffic_ligts_data(conn)
    simulations[session_id] = {
        'port': port,
        'conn': conn,
        'simulation_data': simulation_data
    }

    return jsonify({
        'sessionId': session_id
    })


@app.route('/stop', methods=['POST'])
def stop_simulation() -> Union[tuple[str, int], Response]:
    """
    Stop the simulation
    :return: JSON response
    """
    global simulations
    request_data = request.get_json()
    session_id = request_data.get('session_id')
    if not session_id:
        return "No session ID provided", 400

    conn = get_connection(session_id)

    if not conn:
        return "Session ID not found", 404
    conn.close()
    simulations.pop(session_id)

    return jsonify({
        'status': 'success'
    })


@app.route('/traffic_lights', methods=['GET'])
def get_traffic_lights() -> Union[tuple[str, int], list[dict[str, Any]]]:
    """
    Get the traffic lights of the simulation
    :return: JSON response
    """
    session_id = request.args.get('session_id')
    if not session_id:
        return "No session ID provided", 400

    if session_id not in simulations:
        return "Session ID not found", 404

    connection = get_connection(session_id)

    if connection is None:
        return "Session ID not found", 404

    returned_traffic_lights = get_traffic_ligts_data(connection)

    # connection.close()
    return returned_traffic_lights


def get_traffic_ligts_data(connection: traci.connection.Connection):
    traffic_lights = connection.trafficlight.getIDList()
    returned_traffic_lights = []
    for traffic_light in traffic_lights:
        logics = []
        for logic in list(connection.trafficlight.getAllProgramLogics(traffic_light)):
            logics.append({
                'program_id': logic.programID,
                'type': logic.type,
                'phases': [{
                    'duration': phase.duration,
                    'minDur': phase.minDur,
                    'maxDur': phase.maxDur,
                    'state': phase.state
                } for phase in logic.phases]
            })
        returned_traffic_lights.append({
            'id': traffic_light,
            'current_phase': connection.trafficlight.getPhase(traffic_light),
            'current_phase_duration': connection.trafficlight.getNextSwitch(
                traffic_light) - connection.simulation.getTime(),
            'current_phase_duration_max': connection.trafficlight.getPhaseDuration(traffic_light),
            'program': connection.trafficlight.getProgram(traffic_light),
            'logics': logics
        })
    return returned_traffic_lights


@app.route('/traffic_lights/<tls_id>/phase', methods=['POST'])
def set_next_phase(tls_id: str) -> Union[tuple[str, int], Response]:
    """
    Set the next phase of a traffic light
    :param tls_id: ID of the traffic light
    :return: JSON response
    """
    request_data = request.get_json()
    session_id = request_data.get('session_id')
    if not session_id:
        return "No session ID provided", 400

    if session_id not in simulations:
        return "Session ID not found", 404

    conn = get_connection(session_id)

    if conn is None:
        return "Session ID not found", 404

    current_index = conn.trafficlight.getPhase(tls_id)
    current_logic = conn.trafficlight.getProgram(tls_id)

    tls = list(filter(lambda x: x['id'] == tls_id, simulations[session_id]['simulation_data']))[0]

    found = False
    for logic in tls['logics']:
        if logic['program_id'] == current_logic:
            next_possible_phase = (current_index + 1) % len(logic['phases'])

            logic['currentPhaseIndex'] = next_possible_phase
            conn.trafficlight.setPhase(tls_id, next_possible_phase)
            found = True
            conn.simulationStep()
            break

    if not found:
        return "Traffic light not found", 404

    return jsonify({
        'status': 'success'
    })


@lru_cache(maxsize=None)  # The maxsize=None means the cache can grow without limit
def calculate_all_possible_transitions(current_phase_state):
    def possible_transitions_for_light(light_state):
        """ Determine all possible transitions for a single light with revised sequence. """
        transitions = {
            'r': ['r', 'y'],  # Red can stay red or turn yellow
            'y': ['y', 'g'],  # Yellow can stay yellow or turn green
            'g': ['g', 'r']  # Green can stay green or turn red
        }
        return transitions.get(light_state, [light_state])

    def combine_transitions(states_list):
        """ Recursive function to combine the transitions for all lights. """
        if len(states_list) == 1:
            return [[state] for state in states_list[0]]
        else:
            combined_transitions = []
            for state in states_list[0]:
                for combined_state in combine_transitions(states_list[1:]):
                    combined_transitions.append([state] + combined_state)
            return combined_transitions

    # Generate a list of possible transitions for each light in the current phase state
    possible_transitions_per_light = [possible_transitions_for_light(light) for light in current_phase_state]

    # Combine these transitions to get all possible states
    combined_transitions = combine_transitions(possible_transitions_per_light)

    # Join each combined state list into a string
    return [''.join(state) for state in combined_transitions]


@app.route('/traffic_lights/<tls_id>/switch_program', methods=['POST'])
def switch_program(tls_id: str) -> Union[tuple[str, int], Response]:
    """
    Switch the program of a traffic light
    :param tls_id: ID of the traffic light
    :return: JSON response
    """
    request_data = request.get_json()
    session_id = request_data.get('session_id')
    new_program_id = request_data.get('program_id')
    if not session_id:
        return "No session ID provided", 400

    if session_id not in simulations:
        return "Session ID not found", 404

    conn = get_connection(session_id)

    if conn is None:
        return "Connection not found", 404

    current_index = conn.trafficlight.getPhase(tls_id)
    current_logic = conn.trafficlight.getProgram(tls_id)

    current_phases = None
    new_phases = None

    tls = list(filter(lambda x: x['id'] == tls_id, simulations[session_id]['simulation_data']))[0]

    for logic in tls['logics']:
        if logic['program_id'] == current_logic:
            current_phases = logic['phases']
            continue
        if logic['program_id'] == new_program_id:
            new_phases = logic['phases']

    # Find the next possible legal phase in the new program
    # Assuming that the string length of the phases is the same both of the programs
    # Possible transitions are:
    # red -> yellow -> green -> red

    current_phase_state = current_phases[current_index]['state']  # e.g. 'GrGGGGg'
    next_possible_phases = calculate_all_possible_transitions(current_phase_state)

    for i, phase in enumerate(new_phases):
        if phase['state'] in next_possible_phases:
            conn.trafficlight.setProgram(tls_id, new_program_id)
            conn.trafficlight.setPhase(tls_id, i)
            tls['current_phase'] = i
            tls['program'] = new_program_id

            conn.simulationStep()

            return jsonify({
                'status': 'success'
            })

    return "Failed to switch logic", 500


@app.route('/step', methods=['POST'])
def step_simulation() -> Union[tuple[str, int], Response]:
    """
    Step the simulation forward
    :return: Simulation metrics after stepping
    """
    global simulations
    request_data = request.get_json()
    session_id = request_data.get('session_id')
    if not session_id:
        return "No session ID provided", 400

    conn = get_connection(session_id)

    if not conn:
        return "Session ID not found", 404

    steps = request_data.get('steps')
    if steps and str(steps).isdigit():
        steps = int(steps)
    else:
        steps = 1

    for i in range(steps):
        conn.simulationStep()

    last_vehicle_in_lane = {}
    longest_waiting_time_car_in_lane = {}
    for i, lane in enumerate(conn.trafficlight.getControlledLanes("209")):
        last_vehicle_in_lane[lane] = conn.lane.getLastStepVehicleIDs(lane)
        if last_vehicle_in_lane[lane]:
            longest_waiting_time_car_in_lane[lane] = conn.vehicle.getWaitingTime(last_vehicle_in_lane[lane][-1])


    return jsonify({
        'status': 'success',
        'how_many_cars_passed_the_tls': 'TODO',
        'cars_in_lanes': last_vehicle_in_lane,
        'longest_waiting_time_car_in_lane': longest_waiting_time_car_in_lane
    })


@app.route('/', methods=['GET'])
def health_check():
    """
    Health check endpoint
    :return: JSON response
    """
    return jsonify({
        'status': 'healthy'
    })


if __name__ == "__main__":
    PORT = os.environ.get('PORT', 8080)
    app.run(port=PORT, host='0.0.0.0')
