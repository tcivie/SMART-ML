import uuid
from typing import Optional, Union, Any

import traci

from dockerImage.sumo_sim.utils import find_available_port, calculate_all_possible_transitions
from dockerImage.sumo_sim.utils import update_config
from utils.misc import average_dict_of_dicts_values_by_key


def initialize_vehicles_in_tls(tls_ids_list):
    """
    Initializes the data structure for vehicle information for each TLS ID.
    :param tls_ids_list: List of TLS IDs.
    :return: Dictionary with initialized data for each TLS.
    """
    vehicles_in_tls = {}
    for tls_id in tls_ids_list:
        vehicles_in_tls[tls_id] = {
            "total": 0,
            "lanes": {},
            'longest_waiting_time_car_in_lane': {}
        }
    return vehicles_in_tls


class Simulation:

    def __init__(self, config_path: str, port: int = None, session_id: str = None, is_gui: bool = False,
                 params: dict = None, architecture: str = None):
        self.vehicles_in_tls = None
        self._conn = None
        self._traffic_lights_cache = None

        self._config_path = config_path
        self._is_gui = is_gui
        self.params = params
        self.architecture = architecture

        self._port = port if port is not None else find_available_port()
        self._session_id = session_id if session_id is not None else str(uuid.uuid4())
        self._output_path = config_path.replace('/*.sumocfg', f'/{self._session_id}tripinfo-output.xml')

        update_config(config_file=config_path, simulation_id=self._session_id)

        print(
            f"Starting SUMO {'with GUI' if is_gui else 'in headless mode'} with config {config_path} on port {self._port}")
        if is_gui:
            sumo_args = ["sumo-gui", "-c", config_path]
        else:
            sumo_args = ["sumo", "-c", config_path]

        traci.start(sumo_args, port=port, label=self._session_id)
        self._conn = traci.getConnection(self._session_id)

    def __dict__(self):
        """
        - All TLS IDs
            - Lane names
            - Programs names(ID's)
                - Number of phases
        - Number of parameters (7) TODO: Make this dynamic
        :return:
        """
        tls_ids = self._get_tls_ids_list()
        ret = {"tls": {}, "params_count": 7}
        ret_tls = ret['tls']
        for tls in tls_ids:
            lanes = list(set(self.conn.trafficlight.getControlledLanes(tls)))  # TODO: See why we get duplicates
            lanes.sort()
            programs = self.conn.trafficlight.getAllProgramLogics(tls)
            logics = []
            for logic in programs:
                logics.append({
                    'program_id': logic.programID,
                    'phases': [{
                        'duration': phase.duration,
                        'minDur': phase.minDur,
                        'maxDur': phase.maxDur,
                        'state': phase.state
                    } for phase in logic.phases]
                })
            ret_tls[tls] = {'lanes': lanes, 'programs': logics}

        return ret

        # list_of_tls = self.conn.trafficlight.getIDList()
        # tls_data = {}
        # for tls in list_of_tls:
        #     tls = self._get_specific_traffic_light(tls_id)
        #     tls_data[tls] = {
        #         'lanes': self.conn.trafficlight.getControlledLanes(tls),
        #     }
        #

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

    def reset(self) -> None:
        self.conn.load(['-c', self._config_path])
        self._traffic_lights_cache = None
        self.vehicles_in_tls = None

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

    def switch_traffic_light_program(self, tls_id: str, new_program_id: str, make_step: int = 1, *, forced=False) -> \
            Union[
                bool, dict]:
        """
        Switch the traffic light program to the new program
        :param make_step: If make simulation step at the end of the change
        :param tls_id:
        :param new_program_id:
        :return: Success or not
        """
        current_index = self.conn.trafficlight.getPhase(tls_id)
        current_logic = self.conn.trafficlight.getProgram(tls_id)

        current_phases, new_phases = None, None
        tls = self._get_specific_traffic_light(tls_id)

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
            if phase['state'] in next_possible_phases or forced:
                self.conn.trafficlight.setProgram(tls_id, new_program_id)
                self.conn.trafficlight.setPhase(tls_id, i)
                if make_step > 0:
                    return self.step_simulation(steps=make_step, tls_ids=[tls_id])
                return True

        return False

    def set_traffic_light_phase(self, tls_id: str, make_step: int = 1) -> Union[dict, bool]:
        """
        Set the traffic light to the next possible phase
        :param make_step: If make simulation step at the end of the change
        :param tls_id:
        :return: Success or not
        """
        current_index = self.conn.trafficlight.getPhase(tls_id)
        current_logic = self.conn.trafficlight.getProgram(tls_id)
        tls = self._get_specific_traffic_light(tls_id)

        for logic in tls['logics']:
            if logic['program_id'] == current_logic:
                next_possible_phase = (current_index + 1) % len(logic['phases'])
                self.conn.trafficlight.setPhase(tls_id, next_possible_phase)
                if make_step > 0:
                    return self.step_simulation(steps=make_step, tls_ids=[tls_id])
                return True
        return False

    def _get_specific_traffic_light(self, tls_id: str) -> Optional[dict]:
        """
        Get the traffic light data for a specific traffic light
        :param tls_id:
        :return: Traffic light data
        """
        all_traffic_lights = self.get_traffic_lights_data()
        # for optimization purposes
        tls = next((tls for tls in all_traffic_lights if tls['id'] == tls_id), None)
        return tls

    def _get_tls_ids_list(self, *tls_ids):
        """
        Returns a list of traffic light system IDs to be processed.
        :param tls_ids: Single TLS ID, list of TLS IDs, or None.
        :return: List of TLS IDs.
        """
        tls_id_unpacked = tls_ids[0] if tls_ids else None

        if isinstance(tls_id_unpacked, str):  # Single TLS ID provided
            return [tls_id_unpacked]
        elif isinstance(tls_id_unpacked, list):  # List of TLS IDs provided
            return tls_id_unpacked
        else:  # No specific TLS IDs provided; use all TLS IDs
            return self.conn.trafficlight.getIDList()

    def step_simulation(self, steps: int = 1, tls_ids=None) -> Optional[
        dict[str, Union[int, dict[Any, dict[str, Union[dict[Any, Any], int]]]]]]:
        """
        Step the simulation by the specified number of steps.
        """
        for _ in range(steps):
            self.conn.simulationStep()

        vehicles_in_tls, delta_cars_in_tls, cars_that_left = self._get_tls_statistics(tls_ids)

        return {
            'vehicles_in_tls': vehicles_in_tls,
            'delta_cars_in_tls': delta_cars_in_tls,
            'cars_that_left': cars_that_left,
            'is_ended': self.conn.simulation.getMinExpectedNumber() == 0
        }

    def _get_tls_statistics(self, tls_ids):
        # Check Diff for self.vehicles_in_tls
        last_vehicles_in_tls = 0
        cars_before_step = set()
        if self.vehicles_in_tls:
            for tls_id in self.vehicles_in_tls:
                last_vehicles_in_tls += self.vehicles_in_tls[tls_id]['total']
                for lane in self.vehicles_in_tls[tls_id]['lanes']:
                    cars_before_step.update(self.vehicles_in_tls[tls_id]['lanes'][lane])

        # Determine the TLS IDs to process
        tls_ids_list = self._get_tls_ids_list(tls_ids)
        self.vehicles_in_tls = initialize_vehicles_in_tls(tls_ids_list)
        for tls_id in tls_ids_list:
            controlled_lanes = self.conn.trafficlight.getControlledLanes(tls_id)
            for lane in controlled_lanes:
                lane_vehicle_ids = self.conn.lane.getLastStepVehicleIDs(lane)
                self.vehicles_in_tls[tls_id]['lanes'][lane] = lane_vehicle_ids
                self.vehicles_in_tls[tls_id]['total'] += len(lane_vehicle_ids)

                metrics_per_lane = self._calculate_lane_metrics(lane, lane_vehicle_ids)
                self.vehicles_in_tls[tls_id]['longest_waiting_time_car_in_lane'][lane] = metrics_per_lane

        # Calculate the delta
        current_vehicles_in_tls = 0
        cars_after_step = set()
        for tls_id in self.vehicles_in_tls:
            current_vehicles_in_tls += self.vehicles_in_tls[tls_id]['total']
            for lane in self.vehicles_in_tls[tls_id]['lanes']:
                cars_after_step.update(self.vehicles_in_tls[tls_id]['lanes'][lane])
        delta_cars_in_tls = current_vehicles_in_tls - last_vehicles_in_tls
        cars_that_left = len(cars_before_step - cars_after_step)
        return self.vehicles_in_tls, delta_cars_in_tls, cars_that_left

    def _calculate_lane_metrics(self, lane, vehicle_ids):
        metrics = {}
        if vehicle_ids:
            max_wait_time = round(self.conn.vehicle.getWaitingTime(vehicle_ids[-1]), 2)
            min_wait_time = round(self.conn.vehicle.getWaitingTime(vehicle_ids[0]), 2)
            total_cars = len(vehicle_ids)

            # Additional metrics
            average_speed = round(self.conn.lane.getLastStepMeanSpeed(lane), 2)
            queue_length = self.conn.lane.getLastStepHaltingNumber(lane)
            occupancy = round(self.conn.lane.getLastStepOccupancy(lane), 2)

            # Emission metrics (example for CO2)
            total_co2_emission = round(sum(self.conn.vehicle.getCO2Emission(veh_id) for veh_id in vehicle_ids), 2)

            metrics.update({
                'max_wait_time': max_wait_time,
                'min_wait_time': min_wait_time,
                'total_cars': total_cars,
                'average_speed': average_speed,
                'queue_length': queue_length,
                'occupancy': occupancy,
                'total_co2_emission': total_co2_emission
            })

        return metrics

    def get_all_sim_data(self):
        ret = dict()
        tls_ids = self._get_tls_ids_list()

        def process_tls(t_id):
            tls_statistics = self.step_simulation(steps=0, tls_ids=t_id)
            ret[t_id] = {'current_time': self.conn.simulation.getCurrentTime(), 'tls_data': {},
                         'vehicle_data': {'average_speed': 0, 'max_wait_time': 0.0, 'min_wait_time': 0.0,
                                          'occupancy': 0, 'queue_length': 0, 'total_cars': 0}}
            data = ret[t_id]
            tls_data = data['tls_data']
            tls_data['logics'] = self.conn.trafficlight.getAllProgramLogics(t_id)
            used_program_id = self.conn.trafficlight.getProgram(t_id)
            for logic in tls_data['logics']:
                if logic.programID != used_program_id:
                    continue
                tls_data['logics'] = [{'state': phase.state, 'duration': phase.duration} for phase in logic.phases]
                break
            tls_data['current_phase_index'] = self.conn.trafficlight.getPhase(t_id)
            vehicle_data = data['vehicle_data']
            vehicles_in_tls = tls_statistics.get('vehicles_in_tls').get(t_id).get('longest_waiting_time_car_in_lane')
            for key in vehicle_data:
                vehicle_data[key] = average_dict_of_dicts_values_by_key(vehicles_in_tls, key)

            ret[t_id] = data

        for tls_id in tls_ids:
            process_tls(tls_id)
        return ret
