import os
import socket
import uuid
from typing import Union, Any, Dict
from flask import Flask, request, jsonify, Response
from dockerImage.sumo_sim.Simulation import Simulation

app = Flask(__name__)
app.debug = False

# Dictionary to keep track of simulations
simulations: Dict[str, Simulation] = {}


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

    sim = Simulation(config_path, is_gui=True)  # TODO: Change to False
    if not sim.conn:
        return jsonify({
            'success': False,
        })

    simulations[sim.session_id] = sim

    return jsonify({
        'sessionId': sim.session_id
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

    sim = simulations.get(session_id)

    if not sim:
        return "Session ID not found", 404
    sim.close()
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

    sim = simulations.get(session_id)

    if sim is None:
        return "Session ID not found", 404

    returned_traffic_lights = sim.get_traffic_lights_data()

    return returned_traffic_lights


@app.route('/traffic_lights/<tls_id>/phase', methods=['POST'])
def set_next_phase(tls_id: str):
    request_data = request.get_json()
    session_id = request_data.get('session_id')
    make_step = request_data.get('make_step', True)

    if not session_id:
        return "No session ID provided", 400

    sim = simulations.get(session_id)
    if not sim:
        return "Session ID not found", 404

    metrics = sim.set_traffic_light_phase(tls_id, make_step=make_step)
    if not metrics:
        return "Traffic light not found or phase update failed", 404

    if make_step:
        return jsonify({'status': 'success',
                        'simulation_metrics': metrics
                        }), 200
    return jsonify({'status': 'success'}), 200


@app.route('/traffic_lights/<tls_id>/switch_program', methods=['POST'])
def switch_program(tls_id: str):
    request_data = request.get_json()
    session_id = request_data.get('session_id')
    new_program_id = request_data.get('program_id')
    make_step = request_data.get('make_step', True)

    if not session_id:
        return "No session ID provided", 400

    sim = simulations.get(session_id)
    if not sim:
        return "Session ID not found", 404

    metrics = sim.switch_traffic_light_program(tls_id, new_program_id, make_step=make_step)
    if not metrics:
        return "Failed to switch logic", 500

    if make_step:
        return jsonify({'status': 'success',
                        'simulation_metrics': metrics
                        }), 200
    return jsonify({'status': 'success'}), 200


@app.route('/step', methods=['POST'])
def step_simulation():
    request_data = request.get_json()
    session_id = request_data.get('session_id')

    if not session_id:
        return "No session ID provided", 400

    sim = simulations.get(session_id)
    if not sim:
        return "Session ID not found", 404

    steps = request_data.get('steps', 1)
    if str(steps).isdigit():
        steps = int(steps)

    metrics = sim.step_simulation(steps=steps, tls_ids='209')
    return jsonify({
        'status': 'success',
        'simulation_metrics': metrics
    }), 200


@app.route('/reset/<session_id>', methods=['POST'])
def reset_simulation(session_id: str):
    sim = simulations.get(session_id)
    if not sim:
        return "Session ID not found", 404
    sim.reset()
    return jsonify({
        'status': 'success'
    }), 200


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
