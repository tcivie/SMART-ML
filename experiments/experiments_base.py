class Experiment:
    def __init__(self, initial_data):
        self.initial_data = initial_data

    def get_all_tls_names(self):
        return list(self.initial_data['data']['tls'].keys())

    @staticmethod
    def get_tls_data(initial_data, tls_id):
        return initial_data['data']['tls'][tls_id]

    @staticmethod
    def get_tls_program_ids(initial_data, tls_id):
        selected_tls = initial_data['data']['tls'][tls_id]
        selected_program_ids = [program['program_id'] for program in selected_tls["programs"]]
        selected_program_ids.sort()
        return selected_program_ids