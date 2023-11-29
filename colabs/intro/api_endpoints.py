import os
import requests

# Base URL of your Flask server
BASE_URL = os.getenv('BASE_URL', 'http://localhost:8080')


def start_simulation(config_path):
    """Starts the simulation and returns the session ID."""
    url = f'{BASE_URL}/start'
    response = requests.post(url, json={'config_path': config_path, 'is_gui': False})
    return response.json().get('sessionId')


def reset_simulation(session_id):
    """Resets the simulation for the given session ID."""
    url = f'{BASE_URL}/reset/{session_id}'
    response = requests.post(url)
    return response.json()


def stop_simulation(session_id):
    """Stops the simulation for the given session ID."""
    url = f'{BASE_URL}/stop'
    response = requests.post(url, json={'session_id': session_id})
    return response.json()


def get_traffic_lights(session_id):
    """Gets the status of all traffic lights for the given session ID."""
    url = f'{BASE_URL}/traffic_lights'
    params = {'session_id': session_id}
    response = requests.get(url, params=params)
    return response.json()


def set_traffic_light_phase(tls_id, session_id, make_step=1):
    """Sets the next traffic light phase for the given traffic light ID."""
    url = f'{BASE_URL}/traffic_lights/{tls_id}/phase'
    response = requests.post(url, json={'session_id': session_id, 'make_step': make_step})
    return response.json()


def switch_traffic_light_program(tls_id, session_id, program_id, make_step=1):
    """Switches the program of a traffic light for the given traffic light ID."""
    url = f'{BASE_URL}/traffic_lights/{tls_id}/switch_program'
    response = requests.post(url, json={'session_id': session_id, 'program_id': program_id, 'make_step': make_step})
    return response.json()


def step_simulation(session_id, steps=1, tls_ids=None):
    """Advances the simulation by the given number of steps."""
    url = f'{BASE_URL}/step'
    response = requests.post(url, json={'session_id': session_id, 'steps': steps, 'show_data_for_tls_ids': tls_ids})
    return response.json()['simulation_metrics']


def get_initial_data(session_id):
    """Gets the initial data for the given session ID."""
    url = f'{BASE_URL}/init_data/{session_id}'
    response = requests.get(url)
    return response.json()


def health_check():
    """Checks the health of the server."""
    url = f'{BASE_URL}/'
    response = requests.get(url)
    return response.json()
