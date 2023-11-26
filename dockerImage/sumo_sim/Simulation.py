import uuid
from typing import Optional

import traci

from dockerImage.sumo_sim.utils import find_available_port, calculate_all_possible_transitions


class Simulation:

    def __init__(self, config_path: str, port: int = None, session_id: str = None, is_gui: bool = False):
        self._conn = None
        self._traffic_lights_cache = None

        self._config_path = config_path
        self._is_gui = is_gui

        self._port = port if port is not None else find_available_port()
        self._session_id = session_id if session_id is not None else str(uuid.uuid4())

        print(
            f"Starting SUMO {'with GUI' if is_gui else 'in headless mode'} with config {config_path} on port {self._port}")
        if is_gui:
            sumo_args = ["sumo-gui", "-c", config_path]
        else:
            sumo_args = ["sumo", "-c", config_path]

        traci.start(sumo_args, port=port, label=self._session_id)
        self._conn = traci.getConnection(self._session_id)

    def __getattr__(self, name):
        if hasattr(self._conn, name):
            return getattr(self._conn, name)
        raise AttributeError(f"'Simulation' object has no attribute '{name}'")

    @property
    def port(self) -> int:
        return self._port

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def conn(self) -> traci.connection.Connection:
        return self._conn

    def close(self) -> None:
        self.conn.close()

    def get_traffic_lights_data(self) -> list[dict]:
        """
        Get the traffic lights data from the simulation(TraCI) and cache it
        :return: List of traffic lights data
        """
        if self._traffic_lights_cache is not None:
            return self._traffic_lights_cache

        traffic_lights = self.conn.trafficlight.getIDList()
        returned_traffic_lights = []
        for traffic_light in traffic_lights:
            logics = []
            for logic in list(self.conn.trafficlight.getAllProgramLogics(traffic_light)):
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
                'current_phase': self.conn.trafficlight.getPhase(traffic_light),
                'current_phase_duration': self.conn.trafficlight.getNextSwitch(
                    traffic_light) - self.conn.simulation.getTime(),
                'current_phase_duration_max': self.conn.trafficlight.getPhaseDuration(traffic_light),
                'program': self.conn.trafficlight.getProgram(traffic_light),
                'logics': logics
            })

        self._traffic_lights_cache = returned_traffic_lights
        return returned_traffic_lights

    def switch_traffic_light_program(self, tls_id: str, new_program_id: str) -> bool:
        """
        Switch the traffic light program to the new program
        :param tls_id:
        :param new_program_id:
        :return: Success or not
        """
        current_index = self.conn.trafficlight.getPhase(tls_id)
        current_logic = self.conn.trafficlight.getProgram(tls_id)

        current_phases, new_phases = None, None
        tls = self.get_specific_traffic_light(tls_id)

        for logic in tls['logics']:
            if logic['program_id'] == current_logic:
                current_phases = logic['phases']
            elif logic['program_id'] == new_program_id:
                new_phases = logic['phases']

        if not new_phases:
            return False

        # Assuming that the string length of the phases is the same in both programs
        # Find the next possible legal phase in the new program
        current_phase_state = current_phases[current_index]['state']
        next_possible_phases = calculate_all_possible_transitions(current_phase_state)

        for i, phase in enumerate(new_phases):
            if phase['state'] in next_possible_phases:
                self.conn.trafficlight.setProgram(tls_id, new_program_id)
                self.conn.trafficlight.setPhase(tls_id, i)
                self.conn.simulationStep()
                return True

        return False

    def set_traffic_light_phase(self, tls_id: str) -> bool:
        """
        Set the traffic light to the next possible phase
        :param tls_id:
        :return: Success or not
        """
        current_index = self.conn.trafficlight.getPhase(tls_id)
        current_logic = self.conn.trafficlight.getProgram(tls_id)
        tls = self.get_specific_traffic_light(tls_id)

        for logic in tls['logics']:
            if logic['program_id'] == current_logic:
                next_possible_phase = (current_index + 1) % len(logic['phases'])
                self.conntrafficlight.setPhase(tls_id, next_possible_phase)
                self.conn.simulationStep()
                return True
        return False

    def get_specific_traffic_light(self, tls_id: str) -> Optional[dict]:
        """
        Get the traffic light data for a specific traffic light
        :param tls_id:
        :return: Traffic light data
        """
        all_traffic_lights = self.get_traffic_lights_data()
        tls = next((tls for tls in all_traffic_lights if tls['id'] == tls_id), None)
        if tls is None:
            return None
        return tls

    def step_simulation(self, steps: int = 1, tls_ids=None) -> dict:
        """
        Step the simulation by the specified number of steps
        :param steps:
        :param tls_ids:
        :return:
        """
        for _ in range(steps):
            self.conn.simulationStep()

        # Determine the TLS IDs to process
        if isinstance(tls_ids, str):  # Single TLS ID provided
            tls_ids_list = [tls_ids]
        elif isinstance(tls_ids, list):  # List of TLS IDs provided
            tls_ids_list = tls_ids
        else:  # No specific TLS IDs provided; use all TLS IDs
            tls_ids_list = self.conn.trafficlight.getIDList()

        last_vehicle_in_lane = {}
        longest_waiting_time_car_in_lane = {}

        for tls_id in tls_ids_list:
            controlled_lanes = self.conn.trafficlight.getControlledLanes(tls_id)
            for lane in controlled_lanes:
                last_vehicle_in_lane[lane] = self.conn.lane.getLastStepVehicleIDs(lane)
                if last_vehicle_in_lane[lane]:
                    longest_waiting_time_car_in_lane[lane] = self.conn.vehicle.getWaitingTime(last_vehicle_in_lane[lane][-1])

        return {
            'cars_in_lanes': last_vehicle_in_lane,
            'longest_waiting_time_car_in_lane': longest_waiting_time_car_in_lane
        }