import math
import os
import sys

import sumolib

# Check for the SUMO_HOME environment variable
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

# This is the command to run SUMO in headless mode
sumoCmd = [
    "sumo-gui",
    "-c", r"C:\Users\omera\Desktop\College\Year 4\Final Project\Cloned\scenarios\bologna\acosta\run.sumocfg"
]

net = sumolib.net.readNet(
    r'C:\Users\omera\Desktop\College\Year 4\Final Project\Cloned\scenarios\bologna\acosta\acosta_buslanes.net.xml')


class Lane:
    def __init__(self, lane_data):
        self._data = lane_data
        self._shape = self._data.getShape()

    def get_min_distance_points(self, other):
        min_distance = math.inf
        point1 = self._shape[0]
        point2 = other._shape[0]

        for my_point in self._shape:
            for other_point in other._shape:
                distance = math.dist(my_point, other_point)
                if distance < min_distance:
                    min_distance = distance
                    point1 = my_point
                    point2 = other_point

        return point1, point2

    def __str__(self):
        return 'Lane: {}'.format(self._data.getID())

    def __repr__(self):
        return self.__str__()


class Link:
    def __init__(self, link_obj):
        self._lane1 = Lane(link_obj[0])
        self._lane2 = Lane(link_obj[1])

    @property
    def get_lane1(self):
        return self._lane1

    @property
    def get_lane2(self):
        return self._lane2

    def get_line_points(self):
        return self.get_lane1.get_min_distance_points(self.get_lane2)

    def __str__(self):
        return 'Link between: {} and {}'.format(self.get_lane1, self.get_lane2)

    def __repr__(self):
        return self.__str__()


class TrafficLight:
    def __init__(self, tls_obj):
        self._obj = tls_obj
        self._connections = [Link(c) for c in self._obj.getConnections()]

    @property
    def connections(self):
        return self._connections.copy()

    def __str__(self):
        return 'TrafficLight: {}'.format(self._obj.getID())

    def __repr__(self):
        return self.__str__()


def do_segments_intersect(link1: Link, link2: Link):
    """Return True if line segments 'p1p2' and 'p3p4' intersect within the rectangle formed by their endpoints."""

    p1, p2 = link1.get_line_points()
    p3, p4 = link2.get_line_points()

    def ccw(A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

    def within_bounds(point, bound1, bound2):
        """Check if the point lies within the bounds defined by bound1 and bound2."""
        return min(bound1[0], bound2[0]) <= point[0] <= max(bound1[0], bound2[0]) and \
            min(bound1[1], bound2[1]) <= point[1] <= max(bound1[1], bound2[1])

    intersection = ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)

    if intersection:
        # Check if the intersection occurs within the bounding rectangle
        for point in [p1, p2, p3, p4]:
            if within_bounds(point, p1, p2) and within_bounds(point, p3, p4):
                return True
    return False


traffic_lights = net.getTrafficLights()
tls1 = TrafficLight(traffic_lights[4])
print(tls1)
line_segments = set()
for l1 in tls1.connections:
    for l2 in tls1.connections:
        if l1 != l2:
            res = do_segments_intersect(l1, l2)
            print(f"{l1} and {l2} do", "" if res else "not", "intersects")
            if not res:
                line_segments.add(l1.get_line_points())
                line_segments.add(l2.get_line_points())

line_segments = list(line_segments)
# line_segments = line_segments[:4]

line_segments = [(f'{i + 1000}', 'blue', [*line_segments[i]]) for i in range(len(line_segments))]
import xml.etree.ElementTree as ET


def create_sumo_additional_file(line_segments, file_name):
    """
    Create a SUMO additional file with line segments and text annotations.
    :param line_segments: A list of tuples with line segment coordinates and colors.
    :param file_name: Name of the output XML file.
    """

    # Create the root element
    additional = ET.Element("additional")

    # Iterate over the line segments and create XML elements
    for i, segment in enumerate(line_segments):
        line_id, color, points = segment
        polyline = ET.SubElement(additional, "poly")
        polyline.set("id", line_id)
        polyline.set("color", color)
        polyline.set("type", 'line')
        polyline.set("lineWidth", "0.5")
        polyline.set("layer", "10")
        polyline.set("shape", " ".join([f"{x},{y}" for x, y in points]))

        # # Add POIs for text annotations at each point
        # for point in points:
        #     poi = ET.SubElement(additional, "line")
        #     poi.set("id", f"{line_id}point{point}".replace(",", "_").replace(" ", "_"))
        #     poi.set("color", color)
        #     poi.set("layer", "10")
        #     poi.set("x", str(point[0]))
        #     poi.set("y", str(point[1]))
        #     poi.set("type", "text")
        #     poi.set("text", f"{point[0]},{point[1]}")

    # Write the XML to a file
    tree = ET.ElementTree(additional)
    tree.write(file_name)


# Example usage
# line_segments = [
#     ("line1", "blue", [(1, 1), (4, 4)]),
#     ("line2", "green", [(2, 3), (5, 1)]),
#     ("line3", "red", [(3, 4), (6, 5)])
# ]

create_sumo_additional_file(line_segments, "../scenarios/bologna/acosta/lines_and_annotations.add.xml")
# import matplotlib.pyplot as plt
#
# # Example list of line segments as tuples
#
#
# # Plot each line segment
#
# # Define a color cycle (you can customize this list with your preferred colors)
# colors = ['r', 'b']  # standard color abbreviations in Matplotlib
#
# # Plot each line segment
# for i, segment in enumerate(line_segments):
#     (p1, p2), (p3, p4) = segment
#     color = colors[i % len(colors)]  # cycle through colors
#     plt.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, marker='o')  # use the same color for each segment
#     plt.plot([p3[0], p4[0]], [p3[1], p4[1]], color=color, marker='o')
#     if i == 6:
#         break
# # Customizing the plot
# plt.title("Line Segments Plot")
# plt.xlabel("X-axis")
# plt.ylabel("Y-axis")
# plt.grid(True)
#
# # Show the plot
# plt.show()
