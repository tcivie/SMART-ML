import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os


def get_filenames_from_path_os(directory_path):
    """
    Returns a list of filenames found in the specified directory using the os module.

    :param directory_path: The path to the directory from which to list filenames.
    :return: A list of filenames in the directory.
    """
    filenames = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]
    return filenames


def extract_filename_from_path_pathlib(path):
    """
    Extracts the filename from a given path using the pathlib module.

    :param path: The full path to the file.
    :return: The filename extracted from the path.
    """
    return Path(path).name


simulations_path = Path(r'../colabs/intro/simulations')
fig, axs = plt.subplots(1, 2, figsize=(12, 5))  # 1 row, 2 columns
for xml_path in get_filenames_from_path_os(simulations_path):
    # Load and parse the XML file
    try:
        tree = ET.parse(simulations_path / xml_path)
        root = tree.getroot()
    except:
        print("XML file could not be loaded/found")
        exit()

    durations = []
    route_lengths = []
    time_losses = []

    for tripinfo in root.findall('tripinfo'):
        time_losses.append(float(tripinfo.get('timeLoss')))
        durations.append(float(tripinfo.get('duration')))
        route_lengths.append(float(tripinfo.get('routeLength')))

    time_losses = np.array(time_losses)
    durations = np.array(durations)
    route_lengths = np.array(route_lengths)


    def print_stats(data):
        # Descriptive Statistics
        mean = np.mean(data)
        median = np.median(data)
        std_dev = np.std(data)
        variance = np.var(data)
        quartiles = np.percentile(data, [25, 50, 75])
        print(f"Mean: {mean}")
        print(f"Median: {median}")
        print(f"Standard Deviation: {std_dev}")
        print(f"Variance: {variance}")
        print(f"Quartiles: {quartiles}")
        print()


    print('time losses:')
    print_stats(time_losses)
    print('durations:')
    print_stats(durations)

    filename = extract_filename_from_path_pathlib(xml_path).removesuffix('tripinfo-output.xml')
    # Create a figure and a set of subplots

    # Overlaying Histograms on the first subplot
    axs[0].hist(time_losses, bins=30, alpha=0.5, label=filename)
    axs[1].hist(durations, bins=30, alpha=0.5, label=filename)
    axs[0].set_title('Time Losses')
    axs[1].set_title('Durations')
    axs[0].set_xlabel('Time Loss')
    axs[0].set_ylabel('Number of cars')
    axs[1].set_xlabel('Duration')
    axs[1].set_ylabel('Number of cars')
    axs[0].legend()

    # Adjust layout
    plt.tight_layout()

# Show the figure
plt.show()
