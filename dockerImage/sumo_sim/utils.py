import socket
from functools import lru_cache


def find_available_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('', 0))
    port = s.getsockname()[1]
    s.close()
    return port


@lru_cache(maxsize=None)  # The maxsize=None means the cache can grow without limit
def calculate_all_possible_transitions(current_phase_state):
    def possible_transitions_for_light(light_state):
        """ Determine all possible transitions for a single light with revised sequence. """
        transitions = {
            'r': ['r', 'y'],  # Red can stay red or turn yellow
            'y': ['y', 'g'],  # Yellow can stay yellow or turn green
            'g': ['g', 'r']  # Green can stay green or turn red
        }
        return transitions.get(light_state, [light_state])

    def combine_transitions(states_list):
        """ Recursive function to combine the transitions for all lights. """
        if len(states_list) == 1:
            return [[state] for state in states_list[0]]
        else:
            combined_transitions = []
            for state in states_list[0]:
                for combined_state in combine_transitions(states_list[1:]):
                    combined_transitions.append([state] + combined_state)
            return combined_transitions

    # Generate a list of possible transitions for each light in the current phase state
    possible_transitions_per_light = [possible_transitions_for_light(light) for light in current_phase_state]

    # Combine these transitions to get all possible states
    combined_transitions = combine_transitions(possible_transitions_per_light)

    # Join each combined state list into a string
    return [''.join(state) for state in combined_transitions]
