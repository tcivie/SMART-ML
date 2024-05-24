from api_endpoints import *
import json


class SimJsonConverter:
    def __init__(self, sim_file_path, *, is_gui=False):
        self.session_key = start_simulation(sim_file_path, is_gui)
        if self.session_key is None:
            raise RuntimeError("Simulation could not start")

    def __sim_running(self, *, steps=5) -> bool:
        step = step_simulation(self.session_key, steps=steps)
        return not step['is_ended']

    def __process(self):
        ret = dict()
        reset_simulation(self.session_key)

        def analyze_state():
            data = all_simulation_data(self.session_key)
            state = {key: {'current_phase_index': data[key]['tls_data']['current_phase_index']} for key in data}
            time_in_ms = 0
            for key in state:
                tls = state[key]
                time_in_ms = round(data[key]['current_time'] / 1000., 2)
                tls.update(data[key]['vehicle_data'])
            return state, time_in_ms

        def init():
            init_state = all_simulation_data(self.session_key)
            logics = ret.setdefault('logics', dict())
            for tls_id in init_state:
                tls = init_state[tls_id]
                tls_data = tls.get('tls_data')
                logics[tls_id] = tls_data['logics']

        init()
        runtime = ret.setdefault('runtime', dict())
        while self.__sim_running():
            state, current_time = analyze_state()
            runtime[str(current_time)] = state
        return ret

    def convert(self, export_to_json=False):
        ret = self.__process()
        if export_to_json:
            with open(self.session_key + ".json", mode='w') as fp:
                json.dump(ret, fp, indent=2)
        return ret

    def __del__(self):
        stop_simulation(self.session_key)
