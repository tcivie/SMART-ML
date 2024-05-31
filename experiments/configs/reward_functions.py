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


def smooth_traffic_flow(states: dict, cars_that_left: int) -> torch.Tensor:
    desired_speed = 30  # Desired speed in km/h
    reward = cars_that_left * 10  # High base reward for cars that left

    for lane in states.values():
        if not lane:
            # No cars in the lane, continue without changing the reward
            continue

        avg_speed_m_s = lane.get('average_speed', 0)
        avg_speed_kmh = avg_speed_m_s * 3.6  # Convert speed from m/s to km/h
        speed_diff = abs(desired_speed - avg_speed_kmh)
        reward -= speed_diff * 2  # Penalty for deviation from desired speed

        if lane.get('max_wait_time', 0.) > 0:
            reward -= (lane['queue_length'] + lane['max_wait_time'] * 0.3)  # Penalty for long wait times
        else:
            reward += avg_speed_kmh * 1.8  # Reward for maintaining higher average speed

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
