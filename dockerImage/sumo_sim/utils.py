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


def update_config(config_file: str, simulation_id: str, params: list[str,str], architecture: str) -> None:
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

    # <experiment_info>
    #         <parameters>
    #             <param key="HIDDEN_SIZE" value="64" />
    #             <param key="LEARNING_RATE" value="1e-1" />
    #             <param key="EPS_START" value="0.9" />
    #             <param key="EPS_END" value="0.05" />
    #             <param key="EPS_DECAY" value="1_000" />
    #             <param key="BATCH_SIZE" value="128" />
    #             <param key="GAMMA" value="0.80" />
    #             <param key="TAU" value="0.005" />
    #             <param key="MEM_SIZE" value="100_000" />
    #             <param key="EPISODES" value="10" />
    #         </parameters>
    #         <architectures>
    #             <architecture name="DQN" file="dqn.py" />
    #         </architectures>
    #     </experiment_info>
    experiment_info_element = root.find('./experiment-info')
    if experiment_info_element is None:
        experiment_info_element = ET.SubElement(root, 'experiment-info')
    parameter_element = ET.SubElement(experiment_info_element, 'parameters')
    for key, value in params:
        add_or_update_element(parameter_element, key, value)

    architectures_element = ET.SubElement(experiment_info_element, 'architectures')
    add_or_update_element(architectures_element, 'architecture', architecture)

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


def parse_summary_xml(file_path):
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        summary_data = []
        for step in root.findall('step'):
            step_data = {
                'time': float(step.get('time')),
                'loaded': int(step.get('loaded')),
                'inserted': int(step.get('inserted')),
                'running': int(step.get('running')),
                'waiting': int(step.get('waiting')),
                'ended': int(step.get('ended')),
                'arrived': int(step.get('arrived')),
                'collisions': int(step.get('collisions')),
                'teleports': int(step.get('teleports')),
                'halting': int(step.get('halting')),
                'stopped': int(step.get('stopped')),
                'meanWaitingTime': float(step.get('meanWaitingTime')),
                'meanTravelTime': float(step.get('meanTravelTime')),
                'meanSpeed': float(step.get('meanSpeed')),
                'meanSpeedRelative': float(step.get('meanSpeedRelative')),
                'duration': int(step.get('duration'))
            }
            summary_data.append(step_data)
        return summary_data
    except ET.ParseError as e:
        return f"Error parsing XML file: {e}"


def parse_tripinfo_xml(file_path):
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        tripinfo_data = []
        for tripinfo in root.findall('tripinfo'):
            trip_data = {
                'id': tripinfo.get('id'),
                'depart': float(tripinfo.get('depart')),
                'departLane': tripinfo.get('departLane'),
                'departPos': float(tripinfo.get('departPos')),
                'departSpeed': float(tripinfo.get('departSpeed')),
                'departDelay': float(tripinfo.get('departDelay')),
                'arrival': float(tripinfo.get('arrival')),
                'arrivalLane': tripinfo.get('arrivalLane'),
                'arrivalPos': float(tripinfo.get('arrivalPos')),
                'arrivalSpeed': float(tripinfo.get('arrivalSpeed')),
                'duration': float(tripinfo.get('duration')),
                'routeLength': float(tripinfo.get('routeLength')),
                'waitingTime': float(tripinfo.get('waitingTime')),
                'waitingCount': int(tripinfo.get('waitingCount')),
                'stopTime': float(tripinfo.get('stopTime')),
                'timeLoss': float(tripinfo.get('timeLoss')),
                'rerouteNo': int(tripinfo.get('rerouteNo')),
                'vType': tripinfo.get('vType'),
                'speedFactor': float(tripinfo.get('speedFactor'))
            }

            # # Extract emissions data if available
            # emissions = tripinfo.find('emissions')
            # if emissions is not None:
            #     trip_data['emissions'] = {emission: float(emissions.get(emission)) for emission in emissions.keys()}

            tripinfo_data.append(trip_data)
        return tripinfo_data
    except ET.ParseError as e:
        return f"Error parsing XML file: {e}"
