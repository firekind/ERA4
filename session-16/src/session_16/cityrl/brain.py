import math
import random
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from session_16.cityrl.logs import Logger
from session_16.cityrl.world import WorldState


class DrivingDQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DrivingDQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        )

    def forward(self, x):
        return self.net(x)


@dataclass
class Experience:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


@dataclass
class BrainUpdateResult:
    score: float
    temperature: float
    done: bool
    all_waypoints_reached: bool


class Brain:
    def __init__(
        self,
        world: WorldState,
        logger: Logger,
        lr: float,
        temperature: float,
        temperature_min: float,
        temperature_decay: float,
        turn_speed: float,
        sharp_turn_speed: float,
        car_speed: float,
        max_consecutive_crashes: int,
        batch_size: int,
        gamma: float,
        tau: float,
    ):
        self.world = world
        self.logger = logger

        input_dim = 9  # 7 sensors + angle_to_target + distance_to_target
        n_actions = 5  # 0: left, 1: straight, 2: right, 3: sharp left, 4: sharp right

        self.policy_net = DrivingDQN(input_dim, n_actions)
        self.target_net = DrivingDQN(input_dim, n_actions)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

        self.memory: deque[Experience] = deque(maxlen=10000)
        self.priority_memory: deque[Experience] = deque(
            maxlen=3000
        )  # prioritized replay: separate buffer for high reward episodes
        self.current_episode_buffer: list[Experience] = []
        self.episode_scores = deque(maxlen=100)

        self.target_pos = (0.0, 0.0)
        self.steps = 0
        self.consecutive_crashes = 0
        self.alive = True
        self.score = 0
        self.prev_dist = None
        self.current_target_idx = 0
        self.targets_reached = 0
        self.current_temp = temperature

        self.temp_min = temperature_min
        self.temp_decay = temperature_decay
        self.turn_speed = turn_speed
        self.sharp_turn_speed = sharp_turn_speed
        self.car_speed = car_speed
        self.max_consecutive_crashes = max_consecutive_crashes
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau

    def update(self) -> BrainUpdateResult | None:
        if not self.world.has_minimum_simulation_state():
            return

        state, _ = self._get_state()
        action = 0
        prev_target_idx = self.current_target_idx

        with torch.no_grad():
            q = self.policy_net(torch.FloatTensor(state).unsqueeze(0))
            probs = F.softmax(q / self.current_temp, dim=-1)
            action = torch.multinomial(probs, 1).item()

        next_state, reward, done = self._step(action)
        self.current_episode_buffer.append(
            Experience(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done,
            )
        )
        self._optimize()

        if self.current_temp > self.temp_min:
            self.current_temp *= self.temp_decay

        if self.current_target_idx != prev_target_idx:
            self.logger.info(
                f"🎯 Target {prev_target_idx + 1} reached! Moving to waypoint {self.current_target_idx + 1}/{self.world.num_waypoints()}"
            )

        for target_param, policy_param in zip(
            self.target_net.parameters(), self.policy_net.parameters()
        ):
            target_param.data.copy_(
                self.tau * policy_param.data + (1.0 - self.tau) * target_param.data
            )

        self.steps += 1

        all_waypoints_reached = False
        if done:
            self._finalize_episode()
            should_reset_position = False

            if self.consecutive_crashes >= self.max_consecutive_crashes:
                self.logger.boo(
                    f"{self.max_consecutive_crashes} consecutive crashes! Resetting to origin..."
                )
                self.consecutive_crashes = 0
                should_reset_position = True

            if not self.alive:
                self.logger.info(
                    f"crash: {self.consecutive_crashes}/{self.max_consecutive_crashes}"
                )
            else:
                if self.targets_reached == self.world.num_waypoints() - 1:
                    self.logger.yay(
                        f"all {self.world.num_waypoints()} waypoints reached!"
                    )
                    should_reset_position = True
                    all_waypoints_reached = True

            priority_size = len(self.priority_memory)
            regular_size = len(self.memory)
            total_mem = priority_size + regular_size
            priority_pct = (priority_size / total_mem * 100) if total_mem > 0 else 0
            success_rate = priority_size / max(total_mem, 1)
            sampling_ratio = 0.3 + (success_rate * 0.4)

            self.logger.info(
                f"score: {self.score:.0f} | mem: {priority_size}P/{regular_size}R ({priority_pct:.1f}) | sample: {sampling_ratio * 100:.0f}%P"
            )

            self.reset()
            if should_reset_position:
                self.world.reset_car_position_and_angle()

        return BrainUpdateResult(
            score=self.score,
            temperature=self.current_temp,
            done=done,
            all_waypoints_reached=all_waypoints_reached,
        )

    def _step(self, action: int) -> tuple[np.ndarray, float, bool]:
        car_pos = self.world.car.position
        map = self.world.current_map
        if map is None:
            raise Exception("failed to get current map: not available")

        turn_speed = 0
        if action == 0:  # left turn
            turn_speed = -self.turn_speed
        elif action == 1:  # straight
            turn_speed = 0
        elif action == 2:  # right turn
            turn_speed = self.turn_speed
        elif action == 3:  # sharp left turn
            turn_speed = -self.sharp_turn_speed
        elif action == 4:  # sharp right turn
            turn_speed = self.sharp_turn_speed
        else:
            raise ValueError(f"invalid action: {action}")

        self.world.update_car_angle(self.world.car.angle + turn_speed)
        rad = math.radians(self.world.car.angle)

        new_x = car_pos[0] + math.cos(rad) * self.car_speed
        new_y = car_pos[1] + math.sin(rad) * self.car_speed
        self.world.update_car_position(new_x, new_y)

        next_state, dist = self._get_state()
        reward = -0.1
        done = False

        if self.world.is_colliding(new_x, new_y):
            reward = -100.0
            done = True
            self.alive = False
        elif dist < 20:
            reward = 100.0
            has_next = self._switch_to_next_target()
            if has_next:
                done = False
                _, new_dist = self._get_state()
                self.prev_dist = new_dist
            else:
                done = True
        else:
            reward += (
                next_state[3] * 20
            )  # if the sensor at the sensor detects the road (value closer to 1) incentivize it.
            if self.prev_dist is not None and dist > self.prev_dist:
                reward -= 10
            self.prev_dist = dist

        self.score += reward
        return next_state, reward, done

    def _optimize(self) -> float:
        total_memory_size = len(self.memory) + len(self.priority_memory)
        if total_memory_size < self.batch_size:
            return 0

        success_rate = len(self.priority_memory) / max(total_memory_size, 1)
        priority_ratio = 0.3 + (success_rate * 0.4)

        priority_samples = int(self.batch_size * priority_ratio)
        regular_samples = self.batch_size - priority_samples

        batch: list[Experience] = []
        if len(self.priority_memory) >= priority_samples:
            batch.extend(random.sample(self.priority_memory, priority_samples))
        else:
            batch.extend(list(self.priority_memory))
            regular_samples += priority_samples - len(self.priority_memory)
        if len(self.memory) >= regular_samples:
            batch.extend(random.sample(self.memory, regular_samples))
        else:
            batch.extend(list(self.memory))

        if len(batch) < self.batch_size // 2:
            return 0

        state = torch.FloatTensor(np.array([e.state for e in batch]))
        action = torch.LongTensor([e.action for e in batch]).unsqueeze(1)
        reward = torch.FloatTensor([e.reward for e in batch]).unsqueeze(1)
        next_state = torch.FloatTensor(np.array([e.next_state for e in batch]))
        done = torch.FloatTensor([e.done for e in batch]).unsqueeze(1)

        q = self.policy_net(state).gather(1, action)
        next_q = self.target_net(next_state).max(1)[0].detach().unsqueeze(1)
        target = reward + self.gamma * next_q * (1 - done)

        loss = F.mse_loss(q, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def _get_state(self) -> tuple[np.ndarray, float]:
        car_pos = self.world.car.position

        map = self.world.current_map
        if map is None:
            raise ValueError("failed to get current map: not available")

        sensors = self.world.car.sensors()
        sensor_vals = [self.world.brightness_at(s.pos[0], s.pos[1]) for s in sensors]

        dx = self.target_pos[0] - car_pos[0]
        dy = self.target_pos[1] - car_pos[1]
        dist = math.sqrt(dx * dx + dy * dy)

        rad_to_target = math.atan2(dy, dx)
        angle_to_target = math.degrees(rad_to_target)

        angle_diff = (angle_to_target - self.world.car.angle) % 360
        if angle_diff > 180:
            angle_diff -= 360

        norm_dist = min(dist / 800.0, 1.0)
        norm_angle = angle_diff / 180.0

        state = sensor_vals + [norm_angle, norm_dist]
        return np.array(state, dtype=np.float32), dist

    def _switch_to_next_target(self) -> bool:
        if self.current_target_idx < len(self.world.waypoints) - 1:
            self.current_target_idx += 1
            self.target_pos = self.world.waypoints[self.current_target_idx]
            self.targets_reached += 1
            self.world.set_active_waypoint(self.current_target_idx)
            return True
        return False

    def _finalize_episode(self):
        if len(self.current_episode_buffer) == 0:
            return

        episode_reward = self.score
        self.episode_scores.append(episode_reward)

        if not self.alive:
            self.consecutive_crashes += 1
        else:
            self.consecutive_crashes = 0

        if episode_reward > 0:
            for exp in self.current_episode_buffer:
                self.priority_memory.append(exp)
        else:
            for exp in self.current_episode_buffer:
                self.memory.append(exp)
        self.current_episode_buffer = []

    def reset(self):
        self.alive = True
        self.score = 0
        self.current_target_idx = 0
        self.world.set_active_waypoint(0)
        self.targets_reached = 0
        if self.world.num_waypoints() > 0:
            self.target_pos = self.world.waypoints[0]
        _, dist = self._get_state()
        self.prev_dist = dist
