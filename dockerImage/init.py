import os
from typing import Union, Tuple, Any, List

import traci
from pathlib import Path
from flask import Flask, request, jsonify
import threading
import uuid
import socket

app = Flask(__name__)

# Dictionary to keep track of simulations
simulations = {}


def start_sumo(config_path: str, port: int) -> None:
    sumoCmd = ['sumo', '-c', config_path, '--remote-port', str(port)]
    traci.start(sumoCmd)


def find_available_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('', 0))
    port = s.getsockname()[1]
    s.close()
    return port


@app.route('/start', methods=['GET'])
def start_simulation():
    config_path = request.args.get('config')
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


@app.route('/traffic_lights', methods=['GET'])
def get_traffic_lights(session_id: str) -> Union[tuple[str, int], tuple[List, int]]:
    if not session_id:
        return "No session ID provided", 400

    port = simulations[session_id]['port']
    traci.init(port=port)
    traffic_light_ids = traci.trafficlight.getIDList()
    traci.close()
    return traffic_light_ids, 200


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


if __name__ == "__main__":
    app.run(port=5000)

# # Retrieve the SUMO configuration file name from the environment variable
# sumo_conf_file = os.getenv('SUMO_CONF_FILE', 'run.sumocfg')
#
# # Construct the full path to the configuration file in /sumo-data directory
# config_path = Path('/sumo-data') / sumo_conf_file
#
# # The command to start SUMO, using the full path to the configuration file
# sumoCmd = ["sumo", "-c", str(config_path)]
#
# # Start SUMO with TraCI
# print("Starting SUMO with TraCI...")
# traci.start(sumoCmd, port=8080)
# print("SUMO started.")
#
# while traci.simulation.getMinExpectedNumber() > 0:
#     traci.simulationStep()
#
# # Close the TraCI connection
# traci.close()
# print("TraCI connection closed.")
