import torch

from experiments import device


def penalize_long_wait_times(states: dict, cars_that_left: int) -> torch.Tensor:
    reward = cars_that_left * 5  # Moderate base reward for cars that left
    for lane in states.values():
        if not lane:
            reward += 5
            continue
        max_wait_time = lane.get('max_wait_time', 0.)
        if max_wait_time > 0:
            reward -= (lane['queue_length'] + max_wait_time * 2)  # Heavy penalty for long wait times
        else:
            reward += lane['average_speed'] * 1.5  # Slightly higher reward for speed
    return torch.tensor(reward, dtype=torch.float32, device=device)


def less_penalties_more_rewards_for_cars_that_left(states: dict, cars_that_left: int) -> torch.Tensor:



def even_traffic_distribution(states: dict, cars_that_left: int) -> torch.Tensor:
    reward = cars_that_left * 10  # Base reward for cars that left
    total_cars = sum(lane.get('total_cars', 0) for lane in states.values())
    num_lanes = len(states)
    avg_cars_per_lane = total_cars / num_lanes if num_lanes else 0

    for lane in states.values():
        if not lane:
            reward += 1
            continue
        cars_in_lane = lane.get('total_cars', 0)
        deviation = abs(cars_in_lane - avg_cars_per_lane)
        reward -= deviation  # Penalize deviation from average
        if lane.get('max_wait_time', 0.) > 0:
            reward -= (lane['queue_length'] + lane['max_wait_time'] * 0.5)
        else:
            reward += lane['average_speed'] * 1.2  # Moderate reward for speed
    return torch.tensor(reward, dtype=torch.float32, device=device)


def environmental_impact(states: dict, cars_that_left: int) -> torch.Tensor:
    reward = cars_that_left * 4  # Base reward for cars that left
    for lane in states.values():
        if not lane:
            reward += 5
            continue
        if lane.get('max_wait_time', 0.) > 0:
            idle_penalty = lane['queue_length'] * 0.2 + lane['max_wait_time'] * 0.4
            reward -= idle_penalty  # Penalty for idle times
        fuel_efficiency = lane.get('average_speed', 0) / max(1, lane.get('occupancy', 1))
        reward += fuel_efficiency * 2  # Reward for fuel efficiency
    return torch.tensor(reward, dtype=torch.float32, device=device)


def speed_safety_balance(states: dict, cars_that_left: int) -> torch.Tensor:
    reward = cars_that_left * 3  # Base reward for cars that left
    for lane in states.values():
        if not lane:
            reward += 5
            continue
        sudden_stops_penalty = lane.get('sudden_stops', 0) * 2
        reward -= sudden_stops_penalty  # Penalty for sudden stops
        if lane.get('max_wait_time', 0.) > 0:
            reward -= (lane['queue_length'] + lane['max_wait_time'] * 0.5)
        else:
            reward += lane['average_speed'] * 1.5  # Higher reward for maintaining speed
    return torch.tensor(reward, dtype=torch.float32, device=device)
