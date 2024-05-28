import datetime
import time
from typing import List
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import base64
from io import BytesIO
from typing import Callable, Type
import time
from typing import Callable, Type
from pathlib import Path

import torch
from matplotlib import pyplot as plt
import xml.etree.ElementTree as ET
from typing import List, Dict

from api_endpoints import start_simulation, reset_simulation, step_simulation, stop_simulation
from experiments.configs import SUMO_SIMULATIONS_BASE_PATH
from experiments.experiments_base import Experiment
from experiments.models.models_base import BaseModel

import lxml.etree as lxml_etree


class ConfigBase:
    def __init__(self,
                 epochs: int,
                 step_size: int,
                 model_class: Type[BaseModel],
                 experiment_class,
                 model_params_func: Callable[[object, str], BaseModel.Params],
                 simulation_run_path: str = 'bologna/acosta/run.sumocfg',
                 reward_func: Callable[[dict, int], torch.Tensor] = None, *, is_gui=False
                 ):
        self.model = model_class
        self.reward_func = reward_func
        self.simulation_id = start_simulation(str(SUMO_SIMULATIONS_BASE_PATH / simulation_run_path), is_gui=is_gui)
        self.epochs = epochs
        self.step_size = step_size

        reset_simulation(self.simulation_id)
        self.state = step_simulation(self.simulation_id, 0)

        self.agents = [
            experiment_class(self.simulation_id,
                             tls_id,
                             model=self.model(
                                 model_params_func(self.state, tls_id)
                             ),
                             reward_func=self.reward_func
                             ) for tls_id in self.state['vehicles_in_tls']]

        self.log = ConfigLogging(**vars(self))

    def run_till_end(self):
        state = self.state
        simulation_id = self.simulation_id
        epochs = self.epochs
        step_size = self.step_size
        results = []
        step = 0
        print(f"Starting simulation with {epochs} epochs and {step_size} step size")
        for epoch in range(epochs):
            ended = False
            while not ended:
                for agent in self.agents:
                    call = agent.step(state)
                    if call:
                        call()
                    else:
                        reset_simulation(simulation_id)
                        state = step_simulation(simulation_id, 0)
                        ended = True
                        break
                if not ended:
                    state = step_simulation(simulation_id, step_size)
                    step += step_size
            self.log.print_epoch(epoch, epochs, step)
            self.log.log_epoch(epoch, step)
            results.append(step)
            step = 0
        stop_simulation(simulation_id)
        self.log.plot_results(results, 'Epochs', 'Total Steps', 'Total Steps per Epoch')
        self.log.summarize_run()
        self.log.convert_to_html()

    def __str__(self):
        agents_html = ''.join(
            [f"<tr><td>{agent.session_id}</td><td>{agent.tls_id}</td><td>{agent.model.__class__.__name__}</td></tr>" for
             agent in self.agents])
        return f"""
        <tr>
            <td>{self.epochs}</td>
            <td>{self.step_size}</td>
            <td>{self.model.__name__}</td>
            <td>{self.simulation_id}</td>
            <td>{self.reward_func.__name__ if self.reward_func else 'None'}</td>
            <td><table>{agents_html}</table></td>
            <td>{self.log.__class__.__name__}</td>
        </tr>
        """

    def __repr__(self):
        return self.__str__()


class ConfigLogging:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.logs = []
        self.unique_id = id(ConfigLogging)
        self.start_time = time.time()

        self.base_path = Path('results')
        self.html_file_path = self.base_path / 'html' / f'simulation_summary_{self.unique_id}.html'
        self.xml_file_path = self.base_path / 'xml' / f'simulation_summary_{self.unique_id}.xml'
        self.plot_file_path = self.base_path / 'plot' / f'plot_{self.unique_id}.png'

    def __str__(self):
        return str(self.kwargs)

    def __repr__(self):
        return 'ConfigLogging'

    def plot_results(self, results: List[int], xlabel: str, ylabel: str, title: str):
        plt.figure(figsize=(12, 6))
        plt.plot(results, marker='o', linestyle='-', color='b')
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(self.plot_file_path)
        plt.show()

    def get_formatted_time(self):
        return datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')

    def print_epoch(self, epoch: int, epochs: int, total_steps: int):
        print(
            f'{self.get_formatted_time()}\t|[{self.unique_id}]| Epoch: [{epoch}/{epochs}]\t| total steps: {total_steps}')

    def format_time_elapsed(self, elapsed_time):
        hours, rem = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(rem, 60)
        return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"

    def log_epoch(self, epoch: int, total_steps: int):
        self.logs.append({
            'timestamp': self.get_formatted_time(),
            'time_elapsed': self.format_time_elapsed(time.time() - self.start_time),
            'epoch': epoch,
            'total_steps': total_steps
        })

    def summarize_run(self):
        root = ET.Element("SimulationSummary")

        params = ET.SubElement(root, "Parameters")
        for key, value in self.kwargs.items():
            param = ET.SubElement(params, key)
            if isinstance(value, List):
                param.text = str(value)[1:-2]
            else:
                param.text = str(value)

        results = ET.SubElement(root, "Results")
        for log in self.logs:
            result = ET.SubElement(results, "Result")
            timestamp = ET.SubElement(result, "TimeStamp")
            timestamp.text = str(log['timestamp'])
            time_elapsed = ET.SubElement(result, "TimeElapsed")
            time_elapsed.text = str(log['time_elapsed'])
            epoch = ET.SubElement(result, "Epoch")
            epoch.text = str(log['epoch'])
            total_steps = ET.SubElement(result, "TotalSteps")
            total_steps.text = str(log['total_steps'])

        tree = ET.ElementTree(root)
        tree.write(self.xml_file_path, encoding='utf-8', xml_declaration=False)

        print(f'Summary written to {self.xml_file_path}')

    def convert_to_html(self):
        # Transform XML to HTML using a simple XSLT or manual transformation
        xslt = '''
        <xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
            <xsl:output method="html"/>
            <xsl:template match="/">
                <html>
                <head>
                    <title>Simulation Summary</title>
                </head>
                <body>
                    <h1>Simulation Summary</h1>
                    <h2>Parameters</h2>
                    <table border="1">
                        <tr>
                            <th>Parameter</th>
                            <th>Value</th>
                        </tr>
                        <xsl:for-each select="SimulationSummary/Parameters/*">
                            <tr>
                                <td><xsl:value-of select="name()"/></td>
                                <td><xsl:value-of select="."/></td>
                            </tr>
                        </xsl:for-each>
                    </table>
                    <h2>Results</h2>
                    <table border="1">
                        <tr>
                            <th>TimeStamp</th>
                            <th>TimeElapsed</th>
                            <th>Epoch</th>
                            <th>Total Steps</th>
                        </tr>
                        <xsl:for-each select="SimulationSummary/Results/Result">
                            <tr>
                                <td><xsl:value-of select="TimeStamp"/></td>
                                <td><xsl:value-of select="TimeElapsed"/></td>
                                <td><xsl:value-of select="Epoch"/></td>
                                <td><xsl:value-of select="TotalSteps"/></td>
                            </tr>
                        </xsl:for-each>
                    </table>
                    <h2>Graph</h2>
                    <img src="data:image/png;base64,{plot_data}" alt="Results Plot"/>
                </body>
                </html>
            </xsl:template>
        </xsl:stylesheet>
        '''

        xml_doc = lxml_etree.parse(self.xml_file_path)
        xslt = xslt.format(plot_data=self._encode_plot_as_base64())
        xslt_doc = lxml_etree.XML(xslt)
        transform = lxml_etree.XSLT(xslt_doc)
        html_doc = transform(xml_doc)

        with open(self.html_file_path, 'wb') as f:
            f.write(lxml_etree.tostring(html_doc, pretty_print=True))

        print(f'HTML summary written to {self.html_file_path}')

    def _encode_plot_as_base64(self) -> str:
        with open(self.plot_file_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
