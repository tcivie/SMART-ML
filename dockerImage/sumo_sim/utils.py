import socket
from functools import lru_cache
import xml.etree.ElementTree as ET


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


def update_config(config_file: str, simulation_id: str) -> None:
    """
    Update the SUMO configuration file to include an output-prefix with the given simulation ID.
    Adds the <output> section if it's missing, along with the necessary sub-elements.

    :param config_file: Path to the SUMO configuration XML file.
    :param simulation_id: Simulation ID to be added as output-prefix.
    """

    # Parse the XML configuration file
    tree = ET.parse(config_file)
    root = tree.getroot()

    # Find or create the 'output' element
    output_element = root.find('./output')
    if output_element is None:
        output_element = ET.SubElement(root, 'output')

    # Add or update the 'output-prefix' element
    add_or_update_element(output_element, 'output-prefix', simulation_id)

    # Ensure other necessary sub-elements are present
    # add_or_update_element(output_element, 'emission-output', 'emission-output.xml') # Really large file - no need for now
    add_or_update_element(output_element, 'summary-output', 'summary.xml')
    add_or_update_element(output_element, 'tripinfo-output', 'tripinfo-output.xml')
    add_or_update_element(output_element, 'statistic-output', 'statistics.xml')

    # Save the updated XML back to the file
    tree.write(config_file, encoding='UTF-8', xml_declaration=True)

def add_or_update_element(parent, tag, value):
    """
    Add a new element or update an existing one within the parent element.

    :param parent: The parent element.
    :param tag: The tag name of the element to add or update.
    :param value: The value to set in the 'value' attribute.
    """
    element = None
    for child in parent:
        if child.tag == tag:
            element = child
            break

    if element is None:
        element = ET.SubElement(parent, tag)
    element.set('value', value)
