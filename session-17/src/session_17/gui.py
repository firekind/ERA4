import logging
import os
import threading
import time
from collections import deque
from dataclasses import dataclass
from queue import Queue
from typing import Any, Callable, Literal, NamedTuple

import cv2
import dacite
import dearpygui.dearpygui as dpg
import numpy as np
import yaml
from numpy.typing import NDArray

import session_17.dpg_utils as dpg_utils
from session_17 import rendering
from session_17.discrete_env import DiscreteTrainer, DiscreteTrainerConfig
from session_17.trainer import Trainer, TrainerStats

# =============================================================================
# Data Structures & Types
# =============================================================================


@dataclass
class Config:
    trainer: Literal["discrete"]
    map: str
    config: dict[str, Any]

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

        return config


# =============================================================================
# Utilities
# =============================================================================


class Clock:
    def __init__(self):
        self._last_time = time.perf_counter()

    def tick(self, fps: int):
        target = 1.0 / fps
        elapsed = time.perf_counter() - self._last_time
        sleep_time = target - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)
        self._last_time = time.perf_counter()


class DearPyGuiLogHandler(logging.Handler):
    def __init__(self, parent_id, max_lines=100):
        super().__init__()
        self.parent_id = parent_id
        self.max_lines = max_lines
        self.log_items = deque(maxlen=max_lines)

        # Create themes for each log level
        self.themes = {
            logging.DEBUG: self._create_theme((150, 150, 150)),
            logging.INFO: self._create_theme((255, 255, 255)),
            logging.WARNING: self._create_theme((255, 255, 0)),
            logging.ERROR: self._create_theme((255, 100, 100)),
            logging.CRITICAL: self._create_theme((255, 0, 0)),
        }

    def _create_theme(self, color):
        with dpg.theme() as theme:
            with dpg.theme_component(dpg.mvText):
                dpg.add_theme_color(dpg.mvThemeCol_Text, color)
        return theme

    def emit(self, record):
        log_entry = self.format(record)

        if len(self.log_items) == self.max_lines:
            dpg.delete_item(self.log_items[0])

        text_id = dpg.add_text(log_entry, parent=self.parent_id)
        dpg.bind_item_theme(text_id, self.themes[record.levelno])
        self.log_items.append(text_id)
        dpg.set_y_scroll(self.parent_id, -1.0)


def show_error_popup(message: str):
    vw = dpg.get_viewport_width()
    vh = dpg.get_viewport_height()

    with dpg.window(
        label="Error",
        modal=True,
        show=True,
        pos=[vw // 2 - 150, vh // 2 - 50],
        no_resize=True,
        autosize=True,
    ):
        dpg.add_text(message, wrap=380)
        dpg.add_spacer(height=10)
        dpg.add_button(
            label="OK",
            width=-1,
            callback=lambda: dpg.delete_item(dpg.get_active_window()),
        )


# =============================================================================
# Result Types
# =============================================================================


class SimFrame(NamedTuple):
    data: NDArray[np.uint8] | None


class SimStepResult(NamedTuple):
    frame: SimFrame
    stats: TrainerStats


PhaseResult = SimFrame | SimStepResult

# =============================================================================
# Simulation Phases
# =============================================================================


class NullPhase:
    def update(self) -> SimFrame:
        return SimFrame(data=np.zeros((0, 0, 4), dtype=np.uint8))


class SetupPhase:
    def __init__(
        self, map: NDArray[np.uint8], config: "Config", max_waypoints: int = 6
    ):
        self.map = map
        self.config = config
        self.start_point: tuple[int, int] | None = None
        self.waypoints: list[tuple[int, int]] = []

        self._max_waypoints = max_waypoints
        self._menu_id: str | int | None = None
        self._mouse_pos: tuple[int, int] | None = None

        self._build_ui()

    def _build_ui(self):
        with dpg.window(
            popup=True,
            show=False,
            autosize=True,
            min_size=(10, 10),
            no_move=False,
            no_background=False,
        ) as menu_id:
            self._menu_id = menu_id
            dpg.add_menu_item(label="Set Start Point", callback=self._on_add_startpoint)
            dpg.add_menu_item(label="Add Waypoint", callback=self._on_add_waypoint)

    def update(self) -> SimFrame:
        frame = self.map.copy()

        if self.start_point is not None:
            rendering.render_car(frame, self.start_point)

        for i, wp in enumerate(self.waypoints):
            rendering.render_waypoint(frame, wp, str(i + 1))

        return SimFrame(data=frame)

    def show_menu(self, mouse_pos: tuple[int, int]):
        assert self._menu_id is not None

        self._mouse_pos = mouse_pos
        dpg.show_item(self._menu_id)

    def _on_add_startpoint(self):
        assert self._mouse_pos is not None
        self.start_point = self._mouse_pos
        self._mouse_pos = None

    def _on_add_waypoint(self):
        assert self._mouse_pos is not None
        if len(self.waypoints) >= self._max_waypoints:
            self._mouse_pos = None
            show_error_popup(f"only {self._max_waypoints} waypoint(s) can be added")
            return

        self.waypoints.append(self._mouse_pos)
        self._mouse_pos = None


class ActivePhase:
    def __init__(
        self,
        map: NDArray[np.uint8],
        config: Config,
        start_point: tuple[int, int],
        waypoints: list[tuple[int, int]],
    ):
        self._trainer: Trainer
        self._running = True

        match config.trainer:
            case "discrete":
                trainer_config = dacite.from_dict(DiscreteTrainerConfig, config.config)
                self._trainer = DiscreteTrainer(
                    map, start_point, waypoints, trainer_config
                )

    def update(self) -> SimStepResult | None:
        if self._running:
            stats, frame = self._trainer.step()
            return SimStepResult(frame=SimFrame(data=frame), stats=stats)

    def pause_simulation(self):
        self._running = False

    def resume_simulation(self):
        self._running = True


Phase = NullPhase | SetupPhase | ActivePhase

# =============================================================================
# UI Components
# =============================================================================


class SimView:
    def __init__(self, on_right_click: Callable[[tuple[int, int]], None] | None = None):
        self._texture_id = None
        self._texture_registry: int | str | None = None
        self._sim_image_id: int | str | None = None
        self._on_right_click = on_right_click

    def build_ui(self):
        self._texture_registry = dpg.add_texture_registry()
        self._texture_id = dpg.add_raw_texture(
            width=0,
            height=0,
            default_value=[0.0] * 4,
            format=dpg.mvFormat_Float_rgba,
            parent=self._texture_registry,
        )
        self._sim_image_id = dpg.add_image(self._texture_id)

        with dpg.item_handler_registry() as image_handler:
            dpg.add_item_clicked_handler(
                button=dpg.mvMouseButton_Right, callback=self._on_image_right_click
            )
        dpg.bind_item_handler_registry(self._sim_image_id, image_handler)

    def update(self, frame: NDArray[np.uint8]):
        assert (
            self._texture_id is not None
            and self._texture_registry is not None
            and self._sim_image_id is not None
        )

        current_height = dpg.get_item_height(self._texture_id)
        current_width = dpg.get_item_width(self._texture_id)

        new_height = frame.shape[0]
        new_width = frame.shape[1]

        # prepping the image data for the texture
        norm_rgb = frame.astype(np.float32) / 255.0
        alpha = np.ones((new_height, new_width, 1), dtype=np.float32)
        rgba = np.concatenate([norm_rgb, alpha], axis=2)
        data = rgba.flatten()
        if data.size == 0:
            data = [0.0] * 4

        # recreate the texture if the dimensions of the image has changed
        if current_height != new_height or current_width != new_width:
            # deleting the current texture
            dpg.delete_item(self._texture_id)
            self._texture_id = None

            # creating a new texture with the new dims
            self._texture_id = dpg.add_raw_texture(
                width=new_width,
                height=new_height,
                default_value=data,  # type: ignore
                format=dpg.mvFormat_Float_rgba,
                parent=self._texture_registry,
            )

            # linking the map item to the texture
            dpg.configure_item(
                self._sim_image_id,
                texture_tag=self._texture_id,
                width=new_width,
                height=new_height,
            )
        else:  # if dimensions are similar, just update the texture data
            dpg.set_value(self._texture_id, data)

    def _on_image_right_click(self):
        assert self._sim_image_id is not None
        if self._on_right_click is None:
            return

        mouse_pos = dpg.get_mouse_pos(local=False)
        image_pos = dpg.get_item_rect_min(self._sim_image_id)

        image_menu_pos = (
            int(mouse_pos[0] - image_pos[0]),
            int(mouse_pos[1] - image_pos[1]),
        )

        self._on_right_click(image_menu_pos)


class ControlPanel:
    def __init__(
        self,
        on_start: Callable[[], bool] | None = None,
        on_pause: Callable[[], None] | None = None,
        on_fps_change: Callable[[int], None] | None = None,
        init_fps: int = 60,
    ):
        self._max_fps = 120  # increasing this causes the sim loop thread to starve other threads, hanging the program
        self._min_fps = 15
        assert self._min_fps <= init_fps <= self._max_fps

        self._on_start = on_start
        self._on_pause = on_pause
        self._on_fps_change = on_fps_change
        self._is_sim_running = False
        self._init_fps = init_fps

        self._start_button: str | int | None = None
        self._fps_slider: str | int | None = None

    def build_ui(self):
        self._start_button = dpg.add_button(
            label="Start / Resume", callback=self._on_start_clicked
        )

        dpg.add_spacer()
        dpg.add_separator()
        dpg.add_spacer(height=5)

        with dpg.table(header_row=False, policy=dpg.mvTable_SizingStretchProp):
            dpg.add_table_column(no_header_label=True)
            dpg.add_table_column(no_header_label=True)

            with dpg.table_row():
                with dpg.group(horizontal=True):
                    dpg.add_text("FPS")
                    dpg.add_spacer()
                with dpg.group(horizontal=True):
                    self._fps_slider = dpg.add_slider_int(
                        min_value=self._min_fps,
                        max_value=self._max_fps,
                        default_value=self._init_fps,
                        width=-1,
                        callback=lambda _, value: self._on_fps_change(value)
                        if self._on_fps_change is not None
                        else None,
                    )

    def reset(self):
        dpg_utils.set_item_label(self._start_button, "Start / Resume")
        dpg_utils.set_value(self._fps_slider, self._init_fps)
        if self._on_fps_change is not None:
            self._on_fps_change(self._init_fps)

        self._is_sim_running = False

    def _on_start_clicked(self, sender: str | int):
        if self._is_sim_running:
            if self._on_pause is not None:
                self._on_pause()
            dpg.set_item_label(sender, "Start / Resume")
            self._is_sim_running = False
        else:
            if self._on_start is not None and not self._on_start():
                return
            dpg.set_item_label(sender, "Pause")
            self._is_sim_running = True


class StatsPanel:
    def __init__(self):
        self._step_item_id: str | int | None = None
        self._episode_item_id: str | int | None = None
        self._reward_item_id: str | int | None = None
        self._ema_reward_item_id: str | int | None = None
        self._temperature_item_id: str | int | None = None
        self._reward_history_series: str | int | None = None
        self._reward_history_x_axis: str | int | None = None
        self._reward_history_y_axis: str | int | None = None
        self._ema_reward_history_series: str | int | None = None
        self._ema_reward_history_x_axis: str | int | None = None
        self._ema_reward_history_y_axis: str | int | None = None

        self._ema_alpha = 0.1
        self._reward_history: deque[tuple[int, float]] = deque(maxlen=50)
        self._ema_reward_history: deque[tuple[int, float]] = deque(maxlen=50)

    def build_ui(self):
        with dpg.table(header_row=False):
            dpg.add_table_column(no_header_label=True)
            dpg.add_table_column(no_header_label=True)

            with dpg.table_row():
                dpg.add_text("Step")
                self._step_item_id = dpg.add_text("0")

            with dpg.table_row():
                dpg.add_text("Episode")
                self._episode_item_id = dpg.add_text("0")

            with dpg.table_row():
                dpg.add_text("Temperature")
                self._temperature_item_id = dpg.add_text("0")

        dpg.add_spacer()
        dpg.add_separator()
        dpg.add_spacer(height=5)

        with dpg.table(header_row=False):
            dpg.add_table_column(no_header_label=True)
            dpg.add_table_column(no_header_label=True)

            with dpg.table_row():
                dpg.add_text("Reward")
                self._reward_item_id = dpg.add_text("0")

        with dpg.theme() as plot_theme:
            with dpg.theme_component(dpg.mvPlot):
                dpg.add_theme_style(
                    dpg.mvPlotStyleVar_PlotPadding,
                    0,
                    0,
                    category=dpg.mvThemeCat_Plots,
                )

        with dpg.plot(no_inputs=True, width=-1, height=60):
            dpg.bind_item_theme(dpg.last_item(), plot_theme)
            self._reward_history_x_axis = dpg.add_plot_axis(
                dpg.mvXAxis,
                no_menus=True,
                no_gridlines=True,
                no_tick_labels=True,
            )
            with dpg.plot_axis(
                dpg.mvYAxis,
                no_menus=True,
                no_gridlines=True,
                no_tick_labels=True,
            ) as y_axis:
                self._reward_history_y_axis = y_axis
                self._reward_history_series = dpg.add_line_series([], [])
            dpg.fit_axis_data(self._reward_history_x_axis)
            dpg.fit_axis_data(y_axis)

        dpg.add_spacer(height=10)
        dpg.add_separator()
        dpg.add_spacer(height=5)

        with dpg.table(header_row=False):
            dpg.add_table_column(no_header_label=True)
            dpg.add_table_column(no_header_label=True)

            with dpg.table_row():
                dpg.add_text("Reward (EMA)")
                self._ema_reward_item_id = dpg.add_text("0")

        with dpg.plot(no_inputs=True, width=-1, height=60):
            dpg.bind_item_theme(dpg.last_item(), plot_theme)
            with dpg.theme() as ema_reward_plot_line_theme:
                with dpg.theme_component(dpg.mvLineSeries):
                    dpg.add_theme_color(
                        dpg.mvPlotCol_Line,
                        (223, 110, 55),
                        category=dpg.mvThemeCat_Plots,
                    )
                    dpg.add_theme_style(
                        dpg.mvPlotStyleVar_LineWeight,
                        2,
                        category=dpg.mvThemeCat_Plots,
                    )
            self._ema_reward_history_x_axis = dpg.add_plot_axis(
                dpg.mvXAxis,
                no_menus=True,
                no_gridlines=True,
                no_tick_labels=True,
            )
            with dpg.plot_axis(
                dpg.mvYAxis,
                no_menus=True,
                no_gridlines=True,
                no_tick_labels=True,
            ) as ema_y_axis:
                self._ema_reward_history_y_axis = ema_y_axis
                self._ema_reward_history_series = dpg.add_line_series([], [])
                dpg.bind_item_theme(dpg.last_item(), ema_reward_plot_line_theme)
            dpg.fit_axis_data(self._ema_reward_history_x_axis)
            dpg.fit_axis_data(ema_y_axis)

    def update(self, stats: TrainerStats):
        dpg_utils.set_value(self._step_item_id, f"{stats.step}")
        dpg_utils.set_value(self._episode_item_id, f"{stats.episode}")
        dpg_utils.set_value(self._reward_item_id, f"{stats.reward:.04f}")
        dpg_utils.set_value(self._temperature_item_id, f"{stats.temperature:.04f}")

        self._reward_history.append((stats.step, stats.reward))

        ema: float
        if len(self._reward_history) <= 1:
            ema = stats.reward
        else:
            ema = (
                self._ema_alpha * stats.reward
                + (1.0 - self._ema_alpha) * self._ema_reward_history[-1][1]
            )

        self._ema_reward_history.append((stats.step, ema))
        dpg_utils.set_value(self._ema_reward_item_id, f"{ema:.04f}")
        self._update_reward_graph()

    def reset(self):
        dpg_utils.set_value(self._step_item_id, "0")
        dpg_utils.set_value(self._episode_item_id, "0")
        dpg_utils.set_value(self._reward_item_id, "0")
        dpg_utils.set_value(self._ema_reward_item_id, "0")
        dpg_utils.set_value(self._temperature_item_id, "0")

        self._reward_history.clear()
        self._ema_reward_history.clear()
        self._update_reward_graph()

    def _update_reward_graph(self):
        dpg_utils.configure_item(
            self._reward_history_series,
            x=[float(v[0]) for v in self._reward_history],
            y=[v[1] for v in self._reward_history],
        )
        dpg_utils.fit_axis_data(self._reward_history_x_axis)
        dpg_utils.fit_axis_data(self._reward_history_y_axis)

        dpg_utils.configure_item(
            self._ema_reward_history_series,
            x=[float(v[0]) for v in self._ema_reward_history],
            y=[v[1] for v in self._ema_reward_history],
        )
        dpg_utils.fit_axis_data(self._ema_reward_history_x_axis)
        dpg_utils.fit_axis_data(self._ema_reward_history_y_axis)


class ConfigPanel:
    def build_ui(self):
        pass


class LogPanel:
    def __init__(self):
        self._log_handler: DearPyGuiLogHandler | None = None

    def build_ui(self):
        self._setup_log_handler()

    def _setup_log_handler(self):
        parent = dpg.last_container()
        if self._log_handler is None:
            self._log_handler = DearPyGuiLogHandler(parent)
            logging.getLogger().addHandler(self._log_handler)


class Dashboard:
    def __init__(
        self,
        on_config_load: Callable[[str], None] | None = None,
        on_start: Callable[[], bool] | None = None,
        on_pause: Callable[[], None] | None = None,
        on_hard_reset: Callable | None = None,
        on_sim_view_right_click: Callable[[tuple[int, int]], None] | None = None,
        on_fps_change: Callable[[int], None] | None = None,
    ):
        self._control_panel = ControlPanel(
            on_start=on_start,
            on_pause=on_pause,
            on_fps_change=on_fps_change,
        )
        self._stats_panel = StatsPanel()
        self._config_panel = ConfigPanel()
        self._log_panel = LogPanel()
        self._on_config_load = on_config_load
        self._on_hard_reset = on_hard_reset
        self._sim_view = SimView(on_right_click=on_sim_view_right_click)

    def build_ui(self):
        with dpg.window() as dashboard:
            dpg.set_primary_window(dashboard, True)
            with dpg.menu_bar():
                with dpg.menu(label="File"):
                    with dpg.file_dialog(
                        show=False,
                        width=700,
                        height=400,
                        callback=self._load_config_file,
                    ) as config_file_dialog:
                        dpg.add_file_extension(".yaml")
                        dpg.add_file_extension(".yml")

                    dpg.add_menu_item(
                        label="Open Config",
                        callback=lambda: dpg.show_item(config_file_dialog),
                    )
                with dpg.menu(label="Simulation"):
                    dpg.add_menu_item(
                        label="Hard Reset",
                        callback=self._on_hard_reset_selected,
                    )

            with dpg.child_window(border=False, height=-200):
                with dpg.table(header_row=False, policy=dpg.mvTable_SizingStretchProp):
                    dpg.add_table_column(no_header_label=True, init_width_or_weight=0.2)
                    dpg.add_table_column(no_header_label=True, init_width_or_weight=0.6)
                    dpg.add_table_column(no_header_label=True, init_width_or_weight=0.2)

                    with dpg.table_row():
                        with dpg.group():
                            with dpg.child_window(height=85):
                                self._control_panel.build_ui()
                            dpg.add_spacer()
                            with dpg.child_window():
                                self._config_panel.build_ui()
                        with dpg.child_window():
                            self._sim_view.build_ui()
                        with dpg.child_window():
                            self._stats_panel.build_ui()

            with dpg.child_window(height=-1):
                self._log_panel.build_ui()

    def update_sim_view(self, frame: NDArray[np.uint8]):
        self._sim_view.update(frame)

    def update_stats_view(self, stats: TrainerStats):
        self._stats_panel.update(stats)

    def reset(self):
        self._control_panel.reset()
        self._stats_panel.reset()

    def _load_config_file(self, _, data: dict):
        if self._on_config_load is not None:
            self._on_config_load(data["file_path_name"])

    def _on_hard_reset_selected(self):
        logging.warning("hard resetting simulator")
        if self._on_hard_reset is not None:
            self._on_hard_reset()


# =============================================================================
# Application
# =============================================================================


class App:
    def __init__(self):
        logging.getLogger().setLevel(logging.INFO)
        self._dashboard: Dashboard | None = None

        self._sim_loop_thread: threading.Thread | None = None
        self._is_running = threading.Event()

        self._sim_phase: NullPhase | SetupPhase | ActivePhase = NullPhase()
        self._sim_phase_lock = threading.Lock()
        self._fps: int = 60
        self._fps_lock = threading.Lock()
        self._result_queue: Queue[PhaseResult] = Queue()

    def run(self):
        self._dashboard = Dashboard(
            on_config_load=self._transition_to_setup_phase,
            on_start=self._transition_to_active_phase,
            on_pause=self._on_sim_pause,
            on_hard_reset=self._transition_to_null_phase,
            on_sim_view_right_click=self._on_sim_view_right_click,
            on_fps_change=self._on_fps_change,
        )
        self._dashboard.build_ui()
        self._is_running.set()
        self._sim_loop_thread = threading.Thread(target=self._sim_loop, daemon=True)
        self._sim_loop_thread.start()

        while dpg.is_dearpygui_running():
            while not self._result_queue.empty():
                result = self._result_queue.get()
                match result:
                    case SimFrame():
                        if result.data is not None:
                            self._dashboard.update_sim_view(result.data)
                    case SimStepResult():
                        if result.frame.data is not None:
                            self._dashboard.update_sim_view(result.frame.data)
                        self._dashboard.update_stats_view(result.stats)

            dpg.render_dearpygui_frame()

    def _sim_loop(self):
        clock = Clock()
        while self._is_running.is_set():
            phase = self._current_phase
            match phase:
                case NullPhase():
                    frame = phase.update()
                    self._result_queue.put(frame)
                case SetupPhase():
                    frame = phase.update()
                    self._result_queue.put(frame)
                case ActivePhase():
                    res = phase.update()
                    if res is not None:
                        self._result_queue.put(res)

            clock.tick(self._current_fps)

    def _transition_to_setup_phase(self, config_path: str):
        if not os.path.exists(config_path):
            show_error_popup("config file does not exist")
            return

        config = Config.from_yaml_file(config_path)
        if not os.path.exists(config.map):
            show_error_popup("map file does not exist")
            return

        logging.info("loaded config")
        map = cv2.cvtColor(cv2.imread(config.map), cv2.COLOR_BGR2RGB).astype(np.uint8)
        self._current_phase = SetupPhase(map=map, config=config)

    def _transition_to_active_phase(self) -> bool:
        phase = self._current_phase
        match phase:
            case SetupPhase():
                start_point = phase.start_point
                waypoints = phase.waypoints
                map = phase.map
                config = phase.config

                if start_point is None:
                    show_error_popup(
                        "cannot start simulation when start point is not set"
                    )
                    return False

                if len(waypoints) == 0:
                    show_error_popup("atleast 1 waypoint has to be set")
                    return False

                try:
                    self._current_phase = ActivePhase(
                        map=map,
                        config=config,
                        start_point=start_point,
                        waypoints=waypoints,
                    )
                except Exception as e:
                    show_error_popup(f"failed to initial phase: {e}")
                    return False

                return True

            case ActivePhase():
                phase.resume_simulation()
                return True

            case _:
                show_error_popup("cannot start simulation when setup is not completed")
                return False

    def _transition_to_null_phase(self):
        self._current_phase = NullPhase()
        if self._dashboard is not None:
            self._dashboard.reset()

    def _on_sim_pause(self):
        phase = self._current_phase
        match phase:
            case ActivePhase():
                phase.pause_simulation()

    def _on_sim_view_right_click(self, mouse_pos: tuple[int, int]):
        phase = self._current_phase
        match phase:
            case SetupPhase():
                phase.show_menu(mouse_pos)

    def _on_fps_change(self, value: int):
        self._current_fps = value

    @property
    def _current_phase(self) -> Phase:
        with self._sim_phase_lock:
            return self._sim_phase

    @_current_phase.setter
    def _current_phase(self, value: Phase):
        with self._sim_phase_lock:
            self._sim_phase = value

    @property
    def _current_fps(self) -> int:
        with self._fps_lock:
            return self._fps

    @_current_fps.setter
    def _current_fps(self, value: int):
        with self._fps_lock:
            self._fps = value

    def __enter__(self):
        dpg.create_context()
        dpg.create_viewport(title="City Nav", height=850, width=1300)
        dpg.setup_dearpygui()
        dpg.show_viewport()

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        dpg.destroy_context()
