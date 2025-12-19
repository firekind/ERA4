from dataclasses import dataclass
from typing import Literal

from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QFrame, QTextEdit, QVBoxLayout

from session_16.cityrl.common import Heading1


@dataclass
class LogEntry:
    message: str
    level: Literal["info", "boo", "yay"]


class Logger(QObject):
    message_logged = Signal(LogEntry)

    def __init__(self):
        super().__init__()

    def info(self, message: str):
        self.message_logged.emit(LogEntry(message=message, level="info"))

    def boo(self, message: str):
        self.message_logged.emit(LogEntry(message=message, level="boo"))

    def yay(self, message: str):
        self.message_logged.emit(LogEntry(message=message, level="yay"))


class LogsView(QFrame):
    def __init__(self, logger: Logger):
        super().__init__()

        layout = QVBoxLayout()
        self.setLayout(layout)

        layout.addWidget(Heading1("Logs"))

        self.log_console = QTextEdit()
        self.log_console.setReadOnly(True)
        layout.addWidget(self.log_console)

        logger.message_logged.connect(self._on_message_logged)

    def _on_message_logged(self, message: LogEntry):
        text = message.message
        if message.level == "boo":
            text = f"<font color='#BF616A'>{text}</font>"
        elif message.level == "yay":
            text = f"<font color='#A3BE8C'>{text}</font>"

        self.log_console.append(text)
        sb = self.log_console.verticalScrollBar()
        if sb is not None:
            sb.setValue(sb.maximum())
