import os
from typing import Union, Tuple, Any, List, Dict

import traci
from pathlib import Path
from flask import Flask, request, jsonify, Response
import threading
import uuid
import socket
import subprocess

app = Flask(__name__)
app.debug = True

# Dictionary to keep track of simulations
simulations = {}


def start_sumo(config_path: str, port: int) -> None:
    print(f"Starting SUMO with config {config_path} on port {port}")
    sumo_args = ['sumo', '-c', config_path, '--remote-port', str(port)]
    subprocess.run(sumo_args)




def find_available_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('', 0))
    port = s.getsockname()[1]
    s.close()
    return port


@app.route('/start', methods=['POST'])
def start_simulation():
    request_data = request.get_json()
    config_path = request_data.get('config_path')
    if not config_path:
        return "No config path provided", 400

    session_id = str(uuid.uuid4())
    port = find_available_port()

    thread = threading.Thread(target=start_sumo, args=(config_path, port))
    thread.start()

    simulations[session_id] = {
        'thread': thread,
        'port': port
    }

    return jsonify({
        'sessionId': session_id,
    })


@app.route('/stop', methods=['POST'])
def stop_simulation(id: str) -> Union[tuple[str, int], Response]:
    session_id = request.args.get('sessionId')
    if not session_id:
        return "No session ID provided", 400

    if session_id not in simulations:
        return "Session ID not found", 404

    port = simulations[session_id]['port']
    traci.init(port=port)
    traci.close()
    simulations.pop(session_id)

    return jsonify({
        'status': 'success'
    })


@app.route('/traffic_lights', methods=['GET'])
def get_traffic_lights() -> Union[tuple[str, int], list[dict[str, Any]]]:
    session_id = request.args.get('session_id')
    if not session_id:
        return "No session ID provided", 400

    port = simulations[session_id]['port']
    connection = traci.connect(port=port)

    if connection is None:
        return "Session ID not found", 404

    traffic_lights = connection.trafficlight.getIDList()
    returned_traffic_lights = []
    for traffic_light in traffic_lights:
        logics = []
        for logic in list(connection.trafficlight.getAllProgramLogics(traffic_light)):
            logics.append({
                'program_id': logic.programID,
                'type': logic.type,
                'currentPhaseIndex': logic.currentPhaseIndex,
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
            'current_phase_duration': connection.trafficlight.getNextSwitch(traffic_light) - connection.simulation.getTime(),
            'current_phase_duration_max': connection.trafficlight.getPhaseDuration(traffic_light),
            'program': connection.trafficlight.getProgram(traffic_light),
            'logics': logics
        })

    connection.close()
    return returned_traffic_lights


@app.route('/set_traffic_light_phase', methods=['POST'])
def set_traffic_light_phase():
    session_id = request.args.get('sessionId')
    if not session_id:
        return "No session ID provided", 400

    port = simulations[session_id]['port']
    traci.init(port=port)

    traffic_light_id = request.args.get('trafficLightId')
    if not traffic_light_id:
        return "No traffic light ID provided", 400

    phase = request.args.get('phase')
    if not phase:
        return "No phase provided", 400

    traci.trafficlight.setPhase(traffic_light_id, int(phase))
    traci.close()

    return jsonify({
        'status': 'success'
    })


@app.route('/', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy'
    })


if __name__ == "__main__":
    PORT = os.environ.get('PORT', 8080)
    app.run(port=PORT, host='0.0.0.0')
