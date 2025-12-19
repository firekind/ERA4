import os
from dataclasses import dataclass

import yaml
from PySide6.QtCore import QPointF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPaintEvent, QPen
from PySide6.QtWidgets import (
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from session_16.cityrl.brain import Brain
from session_16.cityrl.common import Heading1, Heading2
from session_16.cityrl.logs import Logger, LogsView
from session_16.cityrl.map import MapView
from session_16.cityrl.recorder import Recorder
from session_16.cityrl.world import WorldState


@dataclass
class Config:
    map: str
    lr: float
    epsilon: float
    turn_speed: int
    sharp_turn_speed: int
    car_speed: int
    max_consecutive_crashes: int
    batch_size: int
    gamma: float
    tau: float
    sensor_dist: int
    record_dir: str | None = None
    car_height: int = 14
    car_width: int = 8

    @staticmethod
    def from_yaml_file(path: str) -> "Config":
        config: Config
        with open(path) as f:
            data = yaml.safe_load(f)
            config = Config(**data)

        if config.map[0] != "/":
            config.map = os.path.join(
                os.path.dirname(path), config.map
            )  # relative to config file

        if config.record_dir is not None and config.record_dir[0] != "/":
            config.record_dir = os.path.join(os.path.dirname(path), config.record_dir)

        return config


class Dashboard(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RL Trainer")
        self.setMinimumSize(1300, 850)

        self.logger = Logger()
        self.world: WorldState | None = None
        self.brain: Brain | None = None
        self.map_view: MapView | None = None
        self.recorder: Recorder | None = None

        self.ticker = QTimer()
        self.ticker.timeout.connect(self._update_loop)
        self.tick_time_ms = 16

        self.layout_ = QHBoxLayout()
        self.layout_.setContentsMargins(5, 5, 5, 5)
        self.setLayout(self.layout_)

        left_panel_layout = QVBoxLayout()

        self.info_view = InfoWidget()
        left_panel_layout.addWidget(self.info_view, 2)

        logs_view = LogsView(self.logger)
        left_panel_layout.addWidget(logs_view, 3)

        self.layout_.addLayout(left_panel_layout, 1)

        self.load_btn = QPushButton("Load Config")
        self.load_btn.clicked.connect(self._on_load_button_clicked)
        self.layout_.addWidget(self.load_btn, 3, alignment=Qt.AlignmentFlag.AlignCenter)

    def _on_load_button_clicked(self):
        f, _ = QFileDialog.getOpenFileName(
            self, "Load Config", "", "Config (*.yaml *.yml)"
        )
        if not f:
            return

        config = Config.from_yaml_file(f)

        self.world = WorldState(
            car_width=config.car_width,
            car_height=config.car_height,
            sensor_dist=config.sensor_dist,
            logger=self.logger,
        )

        self.brain = Brain(
            self.world,
            self.logger,
            lr=config.lr,
            epsilon=config.epsilon,
            turn_speed=config.turn_speed,
            sharp_turn_speed=config.sharp_turn_speed,
            car_speed=config.car_speed,
            max_consecutive_crashes=config.max_consecutive_crashes,
            batch_size=config.batch_size,
            gamma=config.gamma,
            tau=config.tau,
        )

        if config.record_dir is not None:
            self.recorder = Recorder(
                self,
                60,
                os.path.join(
                    config.record_dir, os.path.splitext(os.path.basename(f))[0]
                ),
            )

        map_layout = QVBoxLayout()

        sim_controls = SimulationControls(self.world)
        sim_controls.started.connect(self._on_simulation_start)
        sim_controls.paused.connect(self._on_simulation_paused)
        sim_controls.reset.connect(self._on_simulation_reset)
        map_layout.addWidget(sim_controls)

        self.map_view = MapView(config.map, self.world, self.logger)
        map_layout.addWidget(self.map_view)

        self.layout_.removeWidget(self.load_btn)
        self.load_btn.deleteLater()
        self.layout_.addLayout(map_layout, 3)

    def _update_loop(self):
        if self.map_view is None or self.brain is None:
            return

        res = self.brain.update()
        if res is not None:
            self.info_view.update_chart(res.score)
            self.info_view.update_eps(res.epsilon)
            self.info_view.update_last_reward(res.score)

            if self.recorder is not None:
                self.recorder.capture()
                if res.done:
                    if res.all_waypoints_reached:
                        self.recorder.save()
                    else:
                        self.recorder.clear()

        self.map_view.update_visuals()

    def _on_simulation_start(self):
        if self.map_view is None:
            return

        self.ticker.start(self.tick_time_ms)

    def _on_simulation_paused(self, paused: bool):
        if self.map_view is None:
            return

        if paused:
            self.ticker.stop()
        else:
            self.ticker.start(self.tick_time_ms)

    def _on_simulation_reset(self):
        if self.world is not None:
            self.world.reset()

        if self.brain is not None:
            self.brain.reset()

        if self.recorder is not None:
            self.recorder.reset()

        if self.map_view is not None:
            self.map_view.update_visuals()
        self.ticker.stop()


class SimulationControls(QToolBar):
    started = Signal()
    paused = Signal(bool)
    reset = Signal()

    def __init__(self, world: WorldState):
        super().__init__()

        self.world = world
        self.setStyleSheet("font-size: 12px;")

        self.start_action = self.addAction("Start")
        self.start_action.setToolTip("Start simulation")
        self.start_action.triggered.connect(self._on_start_triggered)

        self.pause_action = self.addAction("Pause")
        self.pause_action.setToolTip("Pause simulation")
        self.pause_action.setCheckable(True)
        self.pause_action.triggered.connect(self._on_pause_triggered)

        reset_action = self.addAction("Reset")
        reset_action.setToolTip("Reset simulation")
        reset_action.triggered.connect(self._on_reset_triggered)

    def _on_start_triggered(self):
        self.started.emit()
        self.start_action.setEnabled(False)

    def _on_pause_triggered(self, checked: bool):
        self.paused.emit(checked)
        if checked:
            self._pause_to_resume()
        else:
            self._resume_to_pause()

    def _on_reset_triggered(self):
        self.reset.emit()
        self.start_action.setEnabled(True)
        self._resume_to_pause()
        self.pause_action.setChecked(False)

    def _pause_to_resume(self):
        self.pause_action.setText("Resume")
        self.pause_action.setToolTip("Resume simulation")

    def _resume_to_pause(self):
        self.pause_action.setText("Pause")
        self.pause_action.setToolTip("Pause simulation")


class InfoWidget(QWidget):
    def __init__(self):
        super().__init__()

        layout = QVBoxLayout()
        self.setLayout(layout)

        layout.addWidget(Heading1("Info"))

        params_layout = QGridLayout()

        params_layout.addWidget(Heading2("epsilon"), 0, 0)
        self.eps_val = QLabel("")
        params_layout.addWidget(self.eps_val, 0, 1)

        params_layout.addWidget(Heading2("last reward"), 1, 0)
        self.last_reward = QLabel("")
        params_layout.addWidget(self.last_reward, 1, 1)

        layout.addLayout(params_layout, 1)

        self.chart = RewardChart()
        layout.addWidget(self.chart, 4)

    def update_chart(self, new_score: float):
        self.chart.update_chart(new_score)

    def update_eps(self, eps: float):
        self.eps_val.setText(f"{eps:.3f}")

    def update_last_reward(self, value: float):
        self.last_reward.setText(f"{value:.3f}")


class RewardChart(QWidget):
    def __init__(self):
        super().__init__()

        self.setMinimumHeight(150)
        self.scores: list[float] = []
        self.max_points = 50
        self.accent_color = QColor("#88C0D0")

    def update_chart(self, new_score: float):
        self.scores.append(new_score)
        if len(self.scores) > self.max_points:
            self.scores.pop(0)
        self.update()

    def paintEvent(self, a0: QPaintEvent | None):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w = self.width()
        h = self.height()

        if len(self.scores) < 2:
            return

        min_val = min(self.scores)
        max_val = max(self.scores)
        if max_val == min_val:
            max_val += 1

        points = []
        step_x = w / (self.max_points - 1)

        for i, score in enumerate(self.scores):
            x = i * step_x
            ratio = (score - min_val) / (max_val - min_val)
            y = h - (ratio * (h * 0.8) + (h * 0.1))
            points.append(QPointF(x, y))

        path = QPainterPath()
        path.moveTo(points[0])
        for p in points[1:]:
            path.lineTo(p)

        pen = QPen(self.accent_color, 2)
        painter.setPen(pen)
        painter.drawPath(path)

        if len(self.scores) >= 2:
            avg_points = []
            window_size = 10

            for i in range(len(self.scores)):
                start_idx = max(0, i - window_size + 1)
                avg_score = sum(self.scores[start_idx : i + 1]) / (i - start_idx + 1)

                x = i * step_x
                ratio = (avg_score - min_val) / (max_val - min_val)
                y = h - (ratio * (h * 0.8) + (h * 0.1))
                avg_points.append(QPointF(x, y))

            if len(avg_points) > 1:
                avg_path = QPainterPath()
                avg_path.moveTo(avg_points[0])
                for p in avg_points[1:]:
                    avg_path.lineTo(p)

                avg_pen = QPen(QColor(255, 215, 0), 3)
                painter.setPen(avg_pen)
                painter.drawPath(avg_path)

        if min_val < 0 and max_val > 0:
            zero_ratio = (0 - min_val) / (max_val - min_val)
            y_zero = h - (zero_ratio * (h * 0.8) + (h * 0.1))
            painter.setPen(QPen(QColor(255, 255, 255, 50), 1, Qt.PenStyle.DashLine))
            painter.drawLine(0, int(y_zero), w, int(y_zero))

        legend_x = 10
        legend_y = 15

        painter.setPen(QPen(self.accent_color, 2))
        painter.drawLine(legend_x, legend_y, legend_x + 20, legend_y)
        painter.setPen(QPen(QColor(200, 200, 200)))
        painter.drawText(legend_x + 25, legend_y + 4, "Raw")

        painter.setPen(QPen(QColor(255, 215, 0), 3))
        painter.drawLine(legend_x + 60, legend_y, legend_x + 80, legend_y)
        painter.setPen(QPen(QColor(200, 200, 200)))
        painter.drawText(legend_x + 85, legend_y + 4, "Avg (10)")
