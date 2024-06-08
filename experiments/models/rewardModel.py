import torch.nn


class RewardModel:
    def __init__(self, model: torch.nn.Module, optimizer, criterion):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.last_state = None

    def __call__(self, metrics, cars_that_left):
        # self.reward_func(metrics, cars_that_left)
        total_occupancy = sum([x.get('occupancy', 0) for x in metrics.values()])
        total_occupancy /= len(metrics)
        if self.last_state is None:
            self.last_state = total_occupancy
            return 0

        total_occupancy_tensor = torch.tensor(total_occupancy, dtype=torch.float32, requires_grad=True).unsqueeze(0)
        reward = self.model(total_occupancy_tensor)
        loss = self.criterion(total_occupancy_tensor, torch.tensor(0, dtype=torch.float32).unsqueeze(0))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.last_state = total_occupancy
        normalized_reward = (reward - 0.5) * 10
        return normalized_reward

    def __name__(self):
        return self.__class__.__name__
