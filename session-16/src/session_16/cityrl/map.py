from PySide6.QtCore import QPoint, QPointF, QRectF, Qt
from PySide6.QtGui import (
    QAction,
    QBrush,
    QColor,
    QFont,
    QImage,
    QPainter,
    QPen,
    QPixmap,
)
from PySide6.QtWidgets import (
    QGraphicsItem,
    QGraphicsScene,
    QGraphicsView,
    QMenu,
    QStyleOptionGraphicsItem,
    QWidget,
)

from session_16.cityrl.logs import Logger
from session_16.cityrl.world import WorldState


class MapView(QGraphicsView):
    def __init__(self, map_file: str, world: WorldState, logger: Logger):
        super().__init__()

        self.logger = logger
        self.world = world

        self.setScene(QGraphicsScene())
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self.show_context_menu)

        map_image = QImage(map_file).convertToFormat(QImage.Format.Format_RGB32)
        self.world.set_current_map(map_image)
        self.scene().clear()
        self.scene().addPixmap(QPixmap.fromImage(map_image))
        self.logger.info("loaded map")

        self.car_graphics_item = CarGraphicsItem(world.car.width, world.car.height)

    def update_visuals(self):
        car_pos = self.world.car.position
        self.car_graphics_item.setPos(QPointF(car_pos[0], car_pos[1]))
        self.car_graphics_item.setRotation(self.world.car.angle)

        active_waypoint = self.world.active_waypoint
        waypoints = [
            item
            for item in self.scene().items()
            if isinstance(item, WaypointGraphicsItem)
        ]
        waypoints.reverse()
        for i, wp in enumerate(waypoints):
            if i == active_waypoint:
                wp.set_active(True)
            else:
                wp.set_active(False)

        sensor_items = [
            item
            for item in self.scene().items()
            if isinstance(item, SensorGraphicsItem)
        ]
        sensor_items.reverse()
        for item, sensor in zip(sensor_items, self.world.car.sensors()):
            x, y = sensor.pos[0], sensor.pos[1]
            item.setPos(QPointF(x, y))
            if self.world.is_colliding(x, y):
                item.set_colliding(True)
            else:
                item.set_colliding(False)

    def show_context_menu(self, pos: QPoint):
        scene_pos = self.mapToScene(pos)
        actions = self._get_context_menu_actions(scene_pos)
        if len(actions) == 0:
            return

        QMenu.exec(actions, self.mapToGlobal(pos), parent=self)

    def _get_context_menu_actions(self, scene_pos: QPointF) -> list[QAction]:
        actions = []
        if self.world.start_point is None:
            set_start = QAction("Set Start Point")
            set_start.triggered.connect(
                lambda: self._on_start_point_ctx_menu_action(scene_pos)
            )
            actions.append(set_start)

        set_waypoint = QAction("Set Way Point")
        set_waypoint.triggered.connect(
            lambda: self._on_waypoint_ctx_menu_action(scene_pos)
        )
        actions.append(set_waypoint)

        return actions

    def _on_start_point_ctx_menu_action(self, pos: QPointF):
        self.world.set_start_point(pos.x(), pos.y())
        self.car_graphics_item.setPos(pos)

        if self.car_graphics_item not in self.scene().items():
            self.scene().addItem(self.car_graphics_item)
            for sensor in self.world.car.sensors():
                item = SensorGraphicsItem()
                item.setPos(QPointF(sensor.pos[0], sensor.pos[1]))
                self.scene().addItem(item)

        self.logger.info(f"added start point at ({pos.x():.0f}, {pos.y():.0f})")

    def _on_waypoint_ctx_menu_action(self, pos: QPointF):
        self.world.add_waypoint(pos.x(), pos.y())
        number = self.world.num_waypoints()
        waypoint_graphics_item = WaypointGraphicsItem(number, pos)
        self.scene().addItem(waypoint_graphics_item)
        self.logger.info(f"added waypoint {number} at ({pos.x():.0f}, {pos.y():.0f})")


class CarGraphicsItem(QGraphicsItem):
    def __init__(self, width: int, height: int):
        super().__init__()
        self.width = width
        self.height = height

        self.setZValue(100)
        self.brush = QBrush(QColor("#88C0D0"))
        self.pen = QPen(Qt.GlobalColor.white, 1)

    def boundingRect(self) -> QRectF:
        return QRectF(-self.width / 2, -self.height / 2, self.width, self.height)

    def paint(
        self,
        painter: QPainter | None,
        option: QStyleOptionGraphicsItem | None,
        widget: QWidget | None = None,
    ):
        if painter is None:
            return

        painter.setBrush(self.brush)
        painter.setPen(self.pen)
        painter.drawRoundedRect(self.boundingRect(), 2, 2)
        painter.setBrush(Qt.GlobalColor.white)
        painter.drawRect(int(self.width / 2) - 2, -3, 2, 6)


class WaypointGraphicsItem(QGraphicsItem):
    def __init__(self, number: int, pos: QPointF, color: QColor | None = None):
        super().__init__()
        self.setZValue(50)
        self.pulse = 0
        self.growing = True
        self.color = color if color else QColor(0, 255, 255)
        self.is_active = False
        self.number = number
        self.setPos(pos)

    def set_active(self, active: bool):
        self.is_active = active
        self.update()

    def set_color(self, color: QColor):
        self.color = color
        self.update()

    def boundingRect(self):
        return QRectF(-20, -20, 40, 40)

    def paint(
        self,
        painter: QPainter | None,
        option: QStyleOptionGraphicsItem | None,
        widget: QWidget | None = None,
    ):
        if painter is None:
            return
        if self.is_active:
            if self.growing:
                self.pulse += 0.5
                if self.pulse > 10:
                    self.growing = False
            else:
                self.pulse -= 0.5
                if self.pulse < 0:
                    self.growing = True

            r = 10 + self.pulse
            painter.setPen(Qt.PenStyle.NoPen)
            outer_color = QColor(self.color)
            outer_color.setAlpha(100)
            painter.setBrush(QBrush(outer_color))
            painter.drawEllipse(QPointF(0, 0), r, r)
            painter.setBrush(QBrush(self.color))
            painter.setPen(QPen(Qt.GlobalColor.white, 2))
            painter.drawEllipse(QPointF(0, 0), 8, 8)
        else:
            dimmed_color = QColor(self.color)
            dimmed_color.setAlpha(120)
            painter.setPen(QPen(Qt.GlobalColor.white, 1))
            painter.setBrush(QBrush(dimmed_color))
            painter.drawEllipse(QPointF(0, 0), 6, 6)

        painter.setPen(QPen(Qt.GlobalColor.white))
        painter.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        painter.drawText(
            QRectF(-10, -10, 20, 20), Qt.AlignmentFlag.AlignCenter, str(self.number)
        )


class SensorGraphicsItem(QGraphicsItem):
    def __init__(self):
        super().__init__()
        self.setZValue(90)
        self.pulse = 0
        self.pulse_speed = 0.3
        self.is_colliding = False

    def set_colliding(self, colliding):
        self.is_colliding = colliding
        self.update()

    def boundingRect(self):
        return QRectF(-4, -4, 8, 8)

    def paint(
        self,
        painter: QPainter | None,
        option: QStyleOptionGraphicsItem | None,
        widget: QWidget | None = None,
    ):
        if painter is None:
            return

        self.pulse += self.pulse_speed
        if self.pulse > 1.0:
            self.pulse = 0

        if not self.is_colliding:
            color = QColor("#A3BE8C")
            outer_alpha = int(150 * (1 - self.pulse))
        else:
            color = QColor("#BF616A")
            outer_alpha = int(200 * (1 - self.pulse))

        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        outer_size = 3 + (2 * self.pulse)
        outer_color = QColor(color)
        outer_color.setAlpha(outer_alpha)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(outer_color))
        painter.drawEllipse(QPointF(0, 0), outer_size, outer_size)

        painter.setBrush(QBrush(color))
        painter.drawEllipse(QPointF(0, 0), 2, 2)
