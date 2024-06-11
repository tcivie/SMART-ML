import torch
import torch.nn as nn

from experiments.models.components.networks import SimpleNetwork, LSTMNetwork


class RewardModel:
    def __init__(self, tls_data=None):
        if tls_data is not None:
            input_size = len(tls_data['lanes']) * 7  # Number of lanes times number of features per lane
            hidden_size = 32
            num_layers = 3
            output_size = 1
            self.model = LSTMNetwork(input_size, hidden_size, num_layers, output_size)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
            self.criterion = nn.MSELoss()
            self.h0 = torch.zeros(num_layers, 1, hidden_size).to('cpu')  # Assuming 'cpu', change if needed
            self.c0 = torch.zeros(num_layers, 1, hidden_size).to('cpu')
        self.last_speed = None

    def __call__(self, metrics, cars_that_left):
        total_speed = sum([x.get('average_speed', 0) * 3.6 for x in metrics.values()])  # Convert m/s to km/h
        average_speed = total_speed / len(metrics) if len(metrics) > 0 else 0
        total_cars = sum([x.get('total_cars', 0) for x in metrics.values()])

        if total_cars == 0:
            return 0

        if self.last_speed is None:
            self.last_speed = average_speed
            return 0

        speed_trend = average_speed - self.last_speed

        speed_target = max(0, speed_trend + 1)
        speed_target_tensor = torch.tensor([[[speed_target]]], dtype=torch.float32)
        
        state_tensor = self.extract_state_tensor(metrics).unsqueeze(0)

        reward, (self.h0, self.c0) = self.model(state_tensor, self.h0, self.c0)

        loss = self.criterion(average_speed, speed_target_tensor)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.last_speed = average_speed

        normalized_reward = reward.item() - 0.5
        return normalized_reward

    def extract_state_tensor(self, metrics):
        features = []

        for lane_id, lane in metrics.items():
            average_speed = lane.get('average_speed', 0) * 3.6
            max_wait_time = lane.get('max_wait_time', 0)
            min_wait_time = lane.get('min_wait_time', 0)
            occupancy = lane.get('occupancy', 0)
            queue_length = lane.get('queue_length', 0)
            total_cars = lane.get('total_cars', 0)
            total_co2_emission = lane.get('total_co2_emission', 0)

            # Append the lane's features to the list
            features.extend([average_speed, max_wait_time, min_wait_time,
                             occupancy, queue_length, total_cars,
                             total_co2_emission])

        # Convert the list of features to a tensor
        data_tensor = torch.tensor(features, dtype=torch.float32, device='cpu')

        return data_tensor.unsqueeze(0)  # Add batch dimension

    def __name__(self):
        return self.__class__.__name__
