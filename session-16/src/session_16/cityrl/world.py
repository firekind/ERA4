import math
import random
from dataclasses import dataclass

from PySide6.QtGui import QColor, QImage

from session_16.cityrl.logs import Logger


@dataclass
class SensorState:
    angle: int
    pos: tuple[float, float]


@dataclass
class CarState:
    width: int
    height: int
    sensor_dist: int
    position: tuple[float, float] = (0, 0)
    angle: float = 0

    def update_position(self, x: float, y: float):
        self.position = (x, y)

    def update_angle(self, angle: float):
        self.angle = angle

    def sensors(self) -> list[SensorState]:
        # 7 sensors: -45°, -30°, -15°, 0°, 15°, 30°, 45°
        angles = [-45, -30, -15, 0, 15, 30, 45]

        sensors = []
        for a in angles:
            rad = math.radians(self.angle + a)
            sx = self.position[0] + math.cos(rad) * self.sensor_dist
            sy = self.position[1] + math.sin(rad) * self.sensor_dist
            sensors.append(SensorState(angle=a, pos=(sx, sy)))

        return sensors


class WorldState:
    def __init__(
        self, car_width: int, car_height: int, sensor_dist: int, logger: Logger
    ):
        self.car = CarState(width=car_width, height=car_height, sensor_dist=sensor_dist)

        self.start_point: tuple[float, float] | None = None
        self.waypoints: list[tuple[float, float]] = []

        self.logger = logger
        self.current_map: QImage | None = None
        self.active_waypoint: int = 0

    def set_current_map(self, map: QImage):
        self.current_map = map

    def is_colliding(self, x: float, y: float) -> bool:
        return self.brightness_at(x, y) < 0.4

    def brightness_at(self, x: float, y: float) -> float:
        if self.current_map is None:
            raise ValueError("cannot calculate brightness: map not available")

        if 0 <= x < self.current_map.width() and 0 <= y < self.current_map.height():
            c = QColor(self.current_map.pixel(int(x), int(y)))
            return ((c.red() + c.green() + c.blue()) / 3.0) / 255.0
        else:
            return 0.0

    def set_start_point(self, x: float, y: float):
        self.start_point = (x, y)
        self.update_car_position(x, y)

    def add_waypoint(self, x: float, y: float):
        self.waypoints.append((x, y))

    def num_waypoints(self) -> int:
        return len(self.waypoints)

    def set_active_waypoint(self, idx: int):
        self.active_waypoint = idx

    def update_car_position(self, x: float, y: float):
        self.car.update_position(x, y)

    def update_car_angle(self, angle: float):
        self.car.update_angle(angle)

    def reset_car_position_and_angle(self):
        if self.start_point is not None:
            self.car.update_position(self.start_point[0], self.start_point[1])
        else:
            self.car.update_position(0, 0)
        self.car.update_angle(random.randint(0, 360))

    def has_minimum_simulation_state(self) -> bool:
        if (
            self.start_point is not None
            and self.current_map is not None
            and self.num_waypoints() >= 1
        ):
            return True
        return False

    def reset(self):
        self.logger.info("resetting world")
        self.reset_car_position_and_angle()
