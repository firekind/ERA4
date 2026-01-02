import logging
import math
import random
from dataclasses import dataclass, field
from typing import Any, Literal, NamedTuple, SupportsFloat

import gymnasium as gym
import gymnasium.spaces as spaces
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from numpy.typing import NDArray
from torch import Tensor

from session_17 import rendering
from session_17.trainer import Trainer, TrainerStats


class Env(gym.Env):
    metadata: dict[str, Any] = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        map: NDArray[np.uint8],
        start_point: tuple[int, int],
        waypoints: list[tuple[int, int]],
        car_width: int,
        car_height: int,
        car_max_speed: float,
        car_max_turn_angle: float,
        sensor_dist: int,
        render_mode: str | None = None,
    ):
        assert render_mode is None or render_mode in self.metadata["render_modes"], (
            "unsupported render mode"
        )

        self.action_space: spaces.Space[NDArray[np.float32]] = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )
        self.observation_space: spaces.Space[NDArray[np.float32]] = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(9,),
            dtype=np.float32,
        )

        self.render_mode = render_mode
        self._map = map
        self._start_point = (float(start_point[0]), float(start_point[1]))
        self._waypoints = waypoints
        self._car_width = car_width
        self._car_height = car_height
        self._car_max_speed = car_max_speed
        self._car_max_turn_angle = car_max_turn_angle
        self._sensor_dist = sensor_dist

        self._car_pos = self._start_point
        self._car_angle = 0.0
        self._target_waypoint_idx = 0
        self._waypoints_reached = 0
        self._prev_dist: float | None = None

    def step(
        self, action: NDArray[np.float32]
    ) -> tuple[NDArray[np.float32], SupportsFloat, bool, bool, dict[str, Any]]:
        speed, turn_angle = action

        # scaling car speed and turn angle
        speed *= self._car_max_speed
        turn_angle *= self._car_max_turn_angle

        # updating car position and rotation
        self._car_angle += turn_angle
        car_angle_rad = math.radians(self._car_angle)
        self._car_pos = (
            self._car_pos[0] + math.cos(car_angle_rad) * speed,
            self._car_pos[1] + math.sin(car_angle_rad) * speed,
        )

        # getting the next state
        next_state, dist = self._get_state()
        reward = -0.1
        done = False

        if self._is_colliding(self._car_pos):
            reward = -100.0
            done = True

            if not done:
                # attempt to recover and navigate back to waypoint 0
                self._target_waypoint_idx = 0

        elif dist < 20:  # car has reached waypoint
            reward = 100.0
            has_next = self._switch_to_next_waypoint()
            if has_next:
                done = False
                self._prev_dist, _ = self._calculate_target_heading()
            else:
                done = True
        else:
            # if the sensor at the center detects the road (value closer to 1), incentivize it.
            reward += self._centered_sensor_reward(next_state)

            # if the car took an action that increased the distance between it and the current
            # targetted waypoint, reduce the reward.
            reward += self._distance_reward(dist)
            self._prev_dist = dist

            reward += self._speed_reward(speed)

        return (
            next_state,
            reward,
            done,
            False,
            dict(
                num_waypoints_reached=self._waypoints_reached,
            ),
        )

    def render(self) -> NDArray[np.uint8] | None:
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[NDArray[np.float32], dict[str, Any]]:
        super().reset(seed=seed, options=options)

        self._car_pos = self._start_point
        self._car_angle = float(random.randint(0, 360))
        self._target_waypoint_idx = 0
        self._waypoints_reached = 0
        self._prev_dist = None

        state, _ = self._get_state()
        return state, {}

    def _get_state(self) -> tuple[NDArray[np.float32], float]:
        # getting the value of a sensor at the sensor locations.
        # the values of a sensor is the brightness of the pixel at the sensor's location.
        sensor_vals = [self._brightness_at(pos) for pos in self._get_sensors_pos()]

        # getting target heading
        dist, angle = self._calculate_target_heading()

        # normalizing distance and angles
        norm_dist = min(dist / self._map_diag(), 1.0)
        norm_angle = angle / 180.0

        return np.array(sensor_vals + [norm_angle, norm_dist], dtype=np.float32), dist

    def _get_sensors_pos(self) -> list[tuple[float, float]]:
        # 7 sensors: -45°, -30°, -15°, 0°, 15°, 30°, 45°
        angles = [-45, -30, -15, 0, 15, 30, 45]

        sensors: list[tuple[float, float]] = []
        for a in angles:
            rad = math.radians(self._car_angle + a)
            sx = self._car_pos[0] + math.cos(rad) * self._sensor_dist
            sy = self._car_pos[1] + math.sin(rad) * self._sensor_dist
            sensors.append((sx, sy))

        return sensors

    def _centered_sensor_reward(self, state: NDArray[np.float32]) -> float:
        return state[3] * 5

    def _distance_reward(self, dist: float) -> float:
        if self._prev_dist is None:
            return 0

        return ((self._prev_dist - dist) / self._car_max_speed) * 15

    def _speed_reward(self, speed: float) -> float:
        return speed * 2.5

    def _is_colliding(self, pos: tuple[float, float]) -> bool:
        return self._brightness_at(pos) < 0.8

    def _brightness_at(self, pos: tuple[float, float]) -> float:
        x, y = pos
        if 0 <= x < self._map_width() and 0 <= y < self._map_height():
            return float(np.mean(self._map[int(y), int(x)]) / 255.0)
        else:
            return 0.0

    def _calculate_target_heading(self) -> tuple[float, float]:
        # calculating distance to target waypoint
        target_waypoint = self._target_waypoint()
        dx = target_waypoint[0] - self._car_pos[0]
        dy = target_waypoint[1] - self._car_pos[1]
        dist = math.sqrt(dx * dx + dy * dy)

        # calculating angle to target waypoint
        angle_to_target = math.degrees(math.atan2(dy, dx))
        angle = (angle_to_target - self._car_angle) % 360  # converting it between 0-360
        if angle > 180:  # converting it between -180 and 180
            angle -= 360

        return dist, angle

    def _switch_to_next_waypoint(self) -> bool:
        if self._target_waypoint_idx < len(self._waypoints) - 1:
            self._target_waypoint_idx += 1
            self._waypoints_reached += 1
            return True
        return False

    def _target_waypoint(self) -> tuple[int, int]:
        return self._waypoints[self._target_waypoint_idx]

    def _map_width(self) -> int:
        return self._map.shape[1]

    def _map_height(self) -> int:
        return self._map.shape[0]

    def _map_diag(self) -> float:
        return math.sqrt(self._map_width() ** 2 + self._map_height() ** 2)

    def _render_frame(self) -> NDArray[np.uint8]:
        frame = self._map.copy()

        rendering.render_car(
            frame,
            (int(self._car_pos[0]), int(self._car_pos[1])),
            self._car_angle,
            width=self._car_width,
            height=self._car_height,
        )

        for i, wp in enumerate(self._waypoints):
            rendering.render_waypoint(
                frame, wp, str(i + 1), i == self._target_waypoint_idx
            )

        for pos in self._get_sensors_pos():
            rendering.render_sensor(
                frame, (int(pos[0]), int(pos[1])), self._is_colliding(pos)
            )

        return frame


class Experience(NamedTuple):
    state: NDArray[np.float32]
    action: NDArray[np.float32]
    reward: float
    next_state: NDArray[np.float32]
    done: bool


class ReplayBuffer:
    def __init__(self, max_size: int = int(1e6)):
        self._max_size = max_size
        self._storage: list[Experience] = []
        self._ptr = 0

    def add(self, experience: Experience):
        if len(self._storage) == self._max_size:
            self._storage[self._ptr] = experience
        else:
            self._storage.append(experience)
        self._ptr = (self._ptr + 1) % self._max_size

    def sample(self, batch_size: int) -> list[Experience]:
        indices = np.random.randint(0, len(self._storage), size=batch_size)
        return [self._storage[i] for i in indices]


class ActorModel(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, max_action: float):
        super().__init__()

        self._max_action = max_action
        self._layer_1 = nn.Sequential(nn.Linear(state_dim, 400), nn.ReLU())
        self._layer_2 = nn.Sequential(nn.Linear(400, 300), nn.ReLU())
        self._layer_3 = nn.Sequential(nn.Linear(300, action_dim), nn.Tanh())

    def forward(self, x: Tensor) -> Tensor:
        x = self._layer_1(x)
        x = self._layer_2(x)
        x = self._layer_3(x)

        return self._max_action * x


class CriticModel(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()

        self._c1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, 1),
        )

        self._c2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, 1),
        )

    def forward(self, state: Tensor, action: Tensor) -> tuple[Tensor, Tensor]:
        x = torch.cat([state, action], dim=1)
        return self._c1(x), self._c2(x)

    def q1(self, state: Tensor, action: Tensor) -> Tensor:
        x = torch.cat([state, action], dim=1)
        return self._c1(x)


class Agent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_action: float,
        batch_size: int = 100,
        discount: float = 0.99,
        tau: float = 5e-3,
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        policy_update_freq: int = 2,
    ):
        self._actor = ActorModel(state_dim, action_dim, max_action).to(_device())
        self._actor_target = ActorModel(state_dim, action_dim, max_action).to(_device())
        self._actor_target.load_state_dict(self._actor.state_dict())
        self._actor_optimizer = optim.Adam(self._actor.parameters())

        self._critic = CriticModel(state_dim, action_dim).to(_device())
        self._critic_target = CriticModel(state_dim, action_dim).to(_device())
        self._critic_target.load_state_dict(self._critic.state_dict())
        self._critic_optimizer = optim.Adam(self._critic.parameters())

        self._replay_buffer = ReplayBuffer()
        self._max_action = max_action
        self._batch_size = batch_size
        self._discount = discount
        self._tau = tau
        self._policy_noise = policy_noise
        self._noise_clip = noise_clip
        self._policy_update_freq = policy_update_freq

        self._num_iter = 0

    def update(self):
        # sample from buffer
        batch = self._replay_buffer.sample(self._batch_size)
        state = torch.Tensor(np.array([e.state for e in batch])).to(
            _device()
        )  # state s
        action = torch.Tensor(np.array([e.action for e in batch])).to(
            _device()
        )  # action a
        reward = (
            torch.Tensor(np.array([e.reward for e in batch])).unsqueeze(1).to(_device())
        )  # reward r
        next_state = torch.Tensor(np.array([e.next_state for e in batch])).to(
            _device()
        )  # next state s'
        done = (
            torch.Tensor(np.array([e.done for e in batch])).unsqueeze(1).to(_device())
        )

        # from next state s', get the next action from the actor model
        with torch.no_grad():
            next_action = self._actor_target(next_state)

        # add gaussian noise to the next action a', and clamp it to a range supported by the environment
        noise = torch.randn_like(next_action).to(_device()) * self._policy_noise
        noise = noise.clamp(-self._noise_clip, self._noise_clip)
        next_action = (next_action + noise).clamp(-self._max_action, self._max_action)

        with torch.no_grad():
            # The two critic targets take each the couple (s’, a’) as input and return two Q-values Qt1(s’,a’) and Qt2(s’,a’) as outputs
            target_q1, target_q2 = self._critic_target(next_state, next_action)

            # get the final target of the two Critic models, which is: Qt = r + γ * min(Qt1, Qt2), where γ is the discount factor
            target_q = reward + (
                (1 - done) * self._discount * torch.min(target_q1, target_q2)
            )

        # the two critic models take each the couple (s, a) as input and return two Q-values Q1(s, a) and Q2(s, a) as outputs
        current_q1, current_q2 = self._critic(state, action)

        # compute the loss coming from the two critic models: Critic Loss = MSE_Loss(Q1(s,a), Qt) + MSE_Loss(Q2(s,a), Qt)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(
            current_q2, target_q
        )

        # backpropagate critic loss
        self._critic_optimizer.zero_grad()
        critic_loss.backward()
        self._critic_optimizer.step()

        # once every two iterations,
        if self._num_iter % self._policy_update_freq == 0:
            # update actor model by performing gradient ascent on the output of the first critic model
            actor_action = self._actor(state)
            actor_loss = -self._critic.q1(state, actor_action).mean()
            self._actor_optimizer.zero_grad()
            actor_loss.backward()
            self._actor_optimizer.step()

            # update weights of the actor target by polyak averaging
            for param, target_param in zip(
                self._actor.parameters(), self._actor_target.parameters()
            ):
                target_param.data.copy_(
                    self._tau * param.data + (1 - self._tau) * target_param.data
                )

            # update weights of the critic target by polyak averaging
            for param, target_param in zip(
                self._critic.parameters(), self._critic_target.parameters()
            ):
                target_param.data.copy_(
                    self._tau * param.data + (1 - self._tau) * target_param.data
                )

        self._num_iter += 1

    def select_action(self, state: NDArray[np.float32]) -> NDArray[np.float32]:
        with torch.no_grad():
            state_tensor = torch.from_numpy(state).unsqueeze(0).to(_device())
            return self._actor(state_tensor).cpu().numpy().flatten()

    def remember(self, experience: Experience):
        self._replay_buffer.add(experience)


@dataclass
class ContinuousEnvConfig:
    map: str
    car_width: int
    car_height: int
    car_max_speed: float
    car_max_turn_angle: float
    sensor_dist: int
    render_mode: Literal["rgb_array"] | None = None
    start_pos: tuple[int, int] | None = None
    waypoints: list[tuple[int, int]] = field(default_factory=list)


@dataclass
class ContinuousAgentConfig:
    batch_size: int = 100
    discount: float = 0.99
    tau: float = 5e-3
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    policy_update_freq: int = 2


@dataclass
class ContinuousTrainerConfig:
    random_action_steps: int = 10000
    exploration_noise: float = 0.1


class ContinuousTrainer(Trainer):
    def __init__(
        self,
        map: NDArray[np.uint8],
        start_point: tuple[int, int],
        waypoints: list[tuple[int, int]],
        env_config: ContinuousEnvConfig,
        agent_config: ContinuousAgentConfig,
        trainer_config: ContinuousTrainerConfig,
    ):
        self.env = Env(
            map=map,
            start_point=start_point,
            waypoints=waypoints,
            car_height=env_config.car_height,
            car_width=env_config.car_width,
            car_max_speed=env_config.car_max_speed,
            car_max_turn_angle=env_config.car_max_turn_angle,
            sensor_dist=env_config.sensor_dist,
            render_mode=env_config.render_mode,
        )

        obs_shape = self.env.observation_space.shape
        if obs_shape is None:
            raise RuntimeError("environment's observation space has no shape")

        action_shape = self.env.action_space.shape
        if action_shape is None:
            raise RuntimeError("environment's action space has no shape")

        max_action = float(self.env.action_space.high[0])  # type: ignore

        self.agent = Agent(
            state_dim=obs_shape[0],
            action_dim=action_shape[0],
            max_action=max_action,
            batch_size=agent_config.batch_size,
            discount=agent_config.discount,
            tau=agent_config.tau,
            policy_noise=agent_config.policy_noise,
            noise_clip=agent_config.noise_clip,
            policy_update_freq=agent_config.policy_update_freq,
        )

        self._random_action_steps = trainer_config.random_action_steps
        self._exploration_noise = trainer_config.exploration_noise
        self._num_steps = 0
        self._num_episodes = 0
        self._done = False
        self._state, _ = self.env.reset()
        self._num_waypoints_reached = 0

    def step(self) -> tuple[TrainerStats, NDArray[np.uint8] | None]:
        action = self._get_action()
        next_state, reward, self._done, _, info = self.env.step(action)
        reward = float(reward)
        self.agent.remember(
            Experience(self._state, action, reward, next_state, self._done)
        )
        self.agent.update()
        self._num_steps += 1

        if self._num_waypoints_reached != info["num_waypoints_reached"]:
            self._num_waypoints_reached = info["num_waypoints_reached"]
            logging.info(f"reached waypoint {self._num_waypoints_reached}")

        if self._done:
            if reward < 0.0:
                logging.debug("crashed! resetting to origin")
            else:
                logging.info("all waypoints reached!")

            self._num_episodes += 1
            self._state, _ = self.env.reset()
            self._done = False
            self._num_waypoints_reached = 0
        else:
            self._state = next_state

        return TrainerStats(
            step=self._num_steps,
            episode=self._num_episodes,
            reward=reward,
        ), self.env.render()

    def _get_action(self) -> NDArray[np.float32]:
        if self._num_steps < self._random_action_steps:
            return self.env.action_space.sample()
        else:
            action = self.agent.select_action(self._state)
            return (
                action
                + np.random.normal(
                    0,
                    self._exploration_noise,
                    size=self.env.action_space.shape[0],  # type: ignore
                )
            ).clip(self.env.action_space.low, self.env.action_space.high)  # type: ignore


def _device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif torch.mps.is_available():
        return "mps"
    else:
        return "cpu"
