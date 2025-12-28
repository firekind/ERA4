import logging
import math
import random
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal, NamedTuple, SupportsFloat

import gymnasium as gym
import gymnasium.spaces as spaces
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from numpy.typing import NDArray

from session_17 import rendering
from session_17.trainer import Trainer, TrainerStats


class Action(Enum):
    left = 0
    straight = 1
    right = 2
    sharp_left = 3
    sharp_right = 4


class Env(gym.Env):
    metadata: dict[str, Any] = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        map: NDArray[np.uint8],
        start_point: tuple[int, int],
        waypoints: list[tuple[int, int]],
        car_width: int,
        car_height: int,
        sensor_dist: int,
        car_speed: float,
        car_turn_speed: float,
        car_sharp_turn_speed: float,
        render_mode: str | None = None,
    ):
        assert render_mode is None or render_mode in self.metadata["render_modes"], (
            "unsupported render mode"
        )

        self.action_space: spaces.Space[np.uint8] = spaces.Discrete(len(Action))
        self.observation_space: spaces.Space[NDArray[np.float32]] = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(9,),
            dtype=np.float32,
        )

        self.map = map
        self.start_point = (float(start_point[0]), float(start_point[1]))
        self.waypoints = waypoints
        self.car_width = car_width
        self.car_height = car_height
        self.sensor_dist = sensor_dist
        self.car_speed = car_speed
        self.car_turn_speed = car_turn_speed
        self.car_sharp_turn_speed = car_sharp_turn_speed
        self.render_mode = render_mode

        self.car_pos = self.start_point
        self.car_angle = 0.0
        self.target_waypoint_idx = 0
        self.waypoints_reached = 0
        self.prev_dist: float | None = None

    def step(
        self, action: np.uint8
    ) -> tuple[NDArray[np.float32], SupportsFloat, bool, bool, dict[str, Any]]:
        act = Action(action)

        turn_speed = 0.0
        match act:
            case Action.left:
                turn_speed = -self.car_turn_speed
            case Action.straight:
                turn_speed = 0.0
            case Action.right:
                turn_speed = self.car_turn_speed
            case Action.sharp_left:
                turn_speed = -self.car_sharp_turn_speed
            case Action.sharp_right:
                turn_speed = self.car_sharp_turn_speed

        # updating car position and rotation
        self.car_angle += turn_speed
        car_angle_rad = math.radians(self.car_angle)
        self.car_pos = (
            self.car_pos[0] + math.cos(car_angle_rad) * self.car_speed,
            self.car_pos[1] + math.sin(car_angle_rad) * self.car_speed,
        )

        # getting the next state
        next_state, dist = self._get_state()
        reward = -0.1
        done = False

        if self._is_colliding(self.car_pos):
            reward = -100.0
            done = True

            if not done:
                # attempt to recover and navigate back to waypoint 0
                self.target_waypoint_idx = 0

        elif dist < 20:  # car has reached waypoint
            reward = 100.0
            has_next = self._switch_to_next_waypoint()
            if has_next:
                done = False
                self.prev_dist, _ = self._calculate_target_heading()
            else:
                done = True
        else:
            # if the sensor at the center detects the road (value closer to 1), incentivize it.
            reward += next_state[3] * 20

            if self.prev_dist is not None and dist > self.prev_dist:
                reward -= 10
            self.prev_dist = dist

        return (
            next_state,
            reward,
            done,
            False,
            dict(
                num_waypoints_reached=self.waypoints_reached,
            ),
        )

    def render(self) -> NDArray[np.uint8] | None:
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[NDArray[np.float32], dict[str, Any]]:
        super().reset(seed=seed, options=options)

        self.car_pos = self.start_point
        self.car_angle = float(random.randint(0, 360))
        self.target_waypoint_idx = 0
        self.waypoints_reached = 0
        self.prev_dist = None

        state, _ = self._get_state()
        return state, {}

    def _get_state(self) -> tuple[NDArray[np.float32], float]:
        # getting the value of a sensor at the sensor locations.
        # the values of a sensor is the brightness of the pixel at the sensor's location.
        sensor_vals = [self._brightness_at(pos) for pos in self._get_sensors_pos()]

        # getting target heading
        dist, angle = self._calculate_target_heading()

        # normalizing distance and angles
        norm_dist = min(dist / 800.0, 1.0)
        norm_angle = angle / 180.0

        return np.array(sensor_vals + [norm_angle, norm_dist]), dist

    def _get_sensors_pos(self) -> list[tuple[float, float]]:
        # 7 sensors: -45°, -30°, -15°, 0°, 15°, 30°, 45°
        angles = [-45, -30, -15, 0, 15, 30, 45]

        sensors: list[tuple[float, float]] = []
        for a in angles:
            rad = math.radians(self.car_angle + a)
            sx = self.car_pos[0] + math.cos(rad) * self.sensor_dist
            sy = self.car_pos[1] + math.sin(rad) * self.sensor_dist
            sensors.append((sx, sy))

        return sensors

    def _is_colliding(self, pos: tuple[float, float]) -> bool:
        return self._brightness_at(pos) < 0.4

    def _brightness_at(self, pos: tuple[float, float]) -> float:
        x, y = pos
        if 0 <= x < self._map_width() and 0 <= y < self._map_height():
            return float(np.mean(self.map[int(y), int(x)]) / 255.0)
        else:
            return 0.0

    def _calculate_target_heading(self) -> tuple[float, float]:
        # calculating distance to target waypoint
        target_waypoint = self._target_waypoint()
        dx = target_waypoint[0] - self.car_pos[0]
        dy = target_waypoint[1] - self.car_pos[1]
        dist = math.sqrt(dx * dx + dy * dy)

        # calculating angle to target waypoint
        angle_to_target = math.degrees(math.atan2(dy, dx))
        angle = (angle_to_target - self.car_angle) % 360  # converting it between 0-360
        if angle > 180:  # converting it between -180 and 180
            angle -= 360

        return dist, angle

    def _switch_to_next_waypoint(self) -> bool:
        if self.target_waypoint_idx < len(self.waypoints) - 1:
            self.target_waypoint_idx += 1
            self.waypoints_reached += 1
            return True
        return False

    def _target_waypoint(self) -> tuple[int, int]:
        return self.waypoints[self.target_waypoint_idx]

    def _map_width(self) -> int:
        return self.map.shape[1]

    def _map_height(self) -> int:
        return self.map.shape[0]

    def _render_frame(self) -> NDArray[np.uint8]:
        frame = self.map.copy()

        rendering.render_car(
            frame,
            (int(self.car_pos[0]), int(self.car_pos[1])),
            self.car_angle,
            width=self.car_width,
            height=self.car_height,
        )

        for i, wp in enumerate(self.waypoints):
            rendering.render_waypoint(
                frame, wp, str(i + 1), i == self.target_waypoint_idx
            )

        for pos in self._get_sensors_pos():
            rendering.render_sensor(
                frame, (int(pos[0]), int(pos[1])), self._is_colliding(pos)
            )

        return frame


class Experience(NamedTuple):
    state: NDArray[np.float32]
    action: np.uint8
    reward: float
    next_state: NDArray[np.float32]
    done: bool


class AgentStats(NamedTuple):
    priority_memory_len: int
    memory_len: int

    @property
    def total_memory_len(self) -> int:
        return self.priority_memory_len + self.memory_len

    @property
    def priority_pct(self) -> float:
        return (
            (self.priority_memory_len / self.total_memory_len * 100)
            if self.total_memory_len > 0
            else 0
        )

    @property
    def success_rate(self) -> float:
        return self.priority_memory_len / max(self.total_memory_len, 1)

    @property
    def sampling_ratio(self) -> float:
        return 0.3 * (self.success_rate * 0.4)


class Agent:
    def __init__(
        self,
        batch_size: int,
        lr: float,
        temp_init: float,
        temp_min: float,
        temp_decay: float,
        gamma: float,
        tau: float,
    ):
        input_dim = 9  # 7 sensors + angle_to_target + distance_to_target
        n_actions = 5  # 0: left, 1: straight, 2: right, 3: sharp left, 4: sharp right

        self.policy_net = DQNModel(input_dim, n_actions)
        self.target_net = DQNModel(input_dim, n_actions)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

        self.memory: deque[Experience] = deque(maxlen=10000)
        self.priority_memory: deque[Experience] = deque(maxlen=3000)
        self.current_episode_buffer: list[Experience] = []

        self.batch_size = batch_size
        self.temp_current = temp_init
        self.temp_min = temp_min
        self.temp_decay = temp_decay
        self.gamma = gamma
        self.tau = tau

    def update(self) -> AgentStats | None:
        memory_size = len(self.memory)
        priority_mem_size = len(self.priority_memory)
        total_mem_size = memory_size + priority_mem_size

        if total_mem_size < self.batch_size:
            return

        success_rate = priority_mem_size / max(total_mem_size, 1)
        priority_ratio = 0.3 + (success_rate * 0.4)
        num_priority_samples = int(self.batch_size * priority_ratio)
        num_regular_samples = self.batch_size - num_priority_samples

        batch: list[Experience] = []
        if priority_mem_size >= num_priority_samples:
            batch.extend(random.sample(self.priority_memory, num_priority_samples))
        else:
            batch.extend(self.priority_memory)
            num_regular_samples += num_priority_samples - priority_mem_size
        if memory_size >= num_regular_samples:
            batch.extend(random.sample(self.memory, num_regular_samples))
        else:
            batch.extend(self.memory)

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

        for target_param, policy_param in zip(
            self.target_net.parameters(), self.policy_net.parameters()
        ):
            target_param.data.copy_(
                self.tau * policy_param.data + (1.0 - self.tau) * target_param.data
            )

        if self.temp_current > self.temp_min:
            self.temp_current *= self.temp_decay

        return AgentStats(priority_memory_len=priority_mem_size, memory_len=memory_size)

    def select_action(self, state: NDArray[np.float32]) -> np.uint8:
        with torch.no_grad():
            q = self.policy_net(torch.FloatTensor(state).unsqueeze(0))
            probs = F.softmax(q / self.temp_current, dim=-1)
            return np.uint8(torch.multinomial(probs, 1).item())

    def remember(self, experience: Experience):
        self.current_episode_buffer.append(experience)

    def finalize_episode(self):
        if len(self.current_episode_buffer) == 0:
            return

        total_reward = sum([e.reward for e in self.current_episode_buffer])
        if total_reward > 0:
            self.priority_memory.extend(self.current_episode_buffer)
        else:
            self.memory.extend(self.current_episode_buffer)
        self.current_episode_buffer = []


class DQNModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQNModel, self).__init__()
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
class DiscreteEnvConfig:
    map: str
    car_width: int
    car_height: int
    sensor_dist: int
    car_speed: float
    car_turn_speed: float
    car_sharp_turn_speed: float
    render_mode: Literal["rgb_array"] | None = None


@dataclass
class DiscreteAgentConfig:
    lr: float
    temperature: float
    temperature_min: float
    temperature_decay: float
    batch_size: int
    gamma: float
    tau: float


class DiscreteTrainer(Trainer):
    def __init__(
        self,
        map: NDArray[np.uint8],
        start_point: tuple[int, int],
        waypoints: list[tuple[int, int]],
        env_config: DiscreteEnvConfig,
        agent_config: DiscreteAgentConfig,
    ):
        self.env = Env(
            map=map,
            start_point=start_point,
            waypoints=waypoints,
            car_height=env_config.car_height,
            car_width=env_config.car_width,
            sensor_dist=env_config.sensor_dist,
            car_speed=env_config.car_speed,
            car_turn_speed=env_config.car_turn_speed,
            car_sharp_turn_speed=env_config.car_sharp_turn_speed,
            render_mode=env_config.render_mode,
        )
        self.agent = Agent(
            batch_size=agent_config.batch_size,
            lr=agent_config.lr,
            temp_init=agent_config.temperature,
            temp_min=agent_config.temperature_min,
            temp_decay=agent_config.temperature_decay,
            gamma=agent_config.gamma,
            tau=agent_config.tau,
        )

        self.num_steps = 0
        self.num_episodes = 0
        self.done = False
        self.state, _ = self.env.reset()
        self.num_waypoints_reached = 0

    def step(self) -> tuple[TrainerStats, NDArray[np.uint8] | None]:
        action = self.agent.select_action(self.state)
        next_state, reward, self.done, _, info = self.env.step(action)
        reward = float(reward)
        self.agent.remember(
            Experience(self.state, action, reward, next_state, self.done)
        )
        stats = self.agent.update()
        if stats is not None:
            logging.debug(
                f"score: {reward:.2f} | mem: {stats.priority_memory_len}P/{stats.memory_len}R ({stats.priority_pct:.1f}) | sample: {stats.sampling_ratio * 100:.0f}%P"
            )
        self.num_steps += 1

        if self.num_waypoints_reached != info["num_waypoints_reached"]:
            self.num_waypoints_reached = info["num_waypoints_reached"]
            logging.info(f"reached waypoint {self.num_waypoints_reached}")

        if self.done:
            if reward < 0.0:
                logging.debug("crashed! resetting to origin")
            else:
                logging.info("all waypoints reached!")

            self.agent.finalize_episode()
            self.num_episodes += 1
            self.state, _ = self.env.reset()
            self.done = False
            self.num_waypoints_reached = 0
        else:
            self.state = next_state

        return TrainerStats(
            step=self.num_steps,
            episode=self.num_episodes,
            reward=reward,
            temperature=self.agent.temp_current,
        ), self.env.render()
