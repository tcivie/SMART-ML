from colabs.intro.api_endpoints import *


class SimJsonConverter:
    def __init__(self, sim_file_path, *, is_gui=False):
        self.session_key = start_simulation(sim_file_path, is_gui)
        if self.session_key is None:
            raise RuntimeError("Simulation could not start")

    def __sim_running(self) -> bool:
        return True

    def __process(self):
        ret: list[dict] = list()

        def analyze_state() -> dict:
            pass

        while self.__sim_running():
            pass
        return ret
