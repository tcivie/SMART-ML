import os
import time
from multiprocessing import Lock
import requests

from sumo_sim.Simulation import LightPhase

# Base URL of your Flask server
BASE_URL = os.getenv('BASE_URL', 'http://127.0.0.1:8080')
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.169 Safari/537.36',
    'Connection': 'keep-alive',
}

sub_lock = Lock()


def start_simulation(config_path, is_gui):
    """Starts the simulation and returns the session ID."""
    url = f'{BASE_URL}/start'
    with sub_lock:
        response = requests.post(url, json={'config_path': config_path, 'is_gui': is_gui},
                                 headers=HEADERS)
    return response.json().get('sessionId')


def reset_simulation(session_id):
    """Resets the simulation for the given session ID."""
    url = f'{BASE_URL}/reset/{session_id}'
    with sub_lock:
        response = requests.post(url, headers=HEADERS)
    return response.json()


def stop_simulation(session_id):
    """Stops the simulation for the given session ID."""
    url = f'{BASE_URL}/stop'
    with sub_lock:
        response = requests.post(url, json={'session_id': session_id}, headers=HEADERS)
    return response.json()


def get_traffic_lights(session_id):
    """Gets the status of all traffic lights for the given session ID."""
    url = f'{BASE_URL}/traffic_lights'
    params = {'session_id': session_id}
    with sub_lock:
        response = requests.get(url, params=params, headers=HEADERS)
    return response.json()


def set_traffic_light_next_phase(tls_id, session_id, make_step=1):
    """Sets the next traffic light phase for the given traffic light ID."""
    url = f'{BASE_URL}/traffic_lights/{tls_id}/phase'
    with sub_lock:
        response = requests.post(url, json={'session_id': session_id, 'make_step': make_step}, headers=HEADERS)
    return response.json()


def switch_traffic_light_program(tls_id, session_id, program_id, make_step=1, *, forced=False):
    """Switches the program of a traffic light for the given traffic light ID."""
    url = f'{BASE_URL}/traffic_lights/{tls_id}/switch_program'
    with sub_lock:
        response = requests.post(url, json={'session_id': session_id, 'program_id': program_id, 'make_step': make_step,
                                            'forced': forced}, headers=HEADERS)
    if response.ok:
        return response.json()
    return {}


def step_simulation(session_id, steps=1, tls_ids=None):
    """Advances the simulation by the given number of steps."""
    url = f'{BASE_URL}/step'
    with sub_lock:
        response = requests.post(url, json={'session_id': session_id, 'steps': steps, 'show_data_for_tls_ids': tls_ids},
                                 headers=HEADERS
                                 )
    return response.json()['simulation_metrics']


def get_initial_data(session_id):
    """Gets the initial data for the given session ID."""
    url = f'{BASE_URL}/init_data/{session_id}'
    with sub_lock:
        response = requests.get(url, headers=HEADERS)
    return response.json()


def all_simulation_data(session_id):
    """Advances the simulation by the given number of steps."""
    url = f'{BASE_URL}/all_data/{session_id}'
    with sub_lock:
        response = requests.get(url)
    return response.json()


def set_traffic_light_phase(tls_id: str, session_id: str, phase: list[LightPhase]):
    """
    Set the phase of a traffic light
    :param tls_id: Traffic light ID
    :param session_id: Session ID
    :param phase: List of LightPhase objects
    :return: If operation was successful
    """
    url = f'{BASE_URL}/traffic_lights/{tls_id}/set_phase'
    with sub_lock:
        response = requests.post(url, json={'session_id': session_id, 'phase': [p.value for p in phase]},
                                 headers=HEADERS)
    return response.json()


def health_check():
    """Checks the health of the server."""
    url = f'{BASE_URL}/'
    with sub_lock:
        response = requests.get(url, headers=HEADERS)
    return response.json()
