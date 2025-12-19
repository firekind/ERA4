from PySide6.QtGui import QFont
from PySide6.QtWidgets import QLabel


class Heading1(QLabel):
    def __init__(self, text: str):
        super().__init__(text)
        font = QFont()
        font.setPointSize(16)
        font.setBold(True)
        self.setFont(font)


class Heading2(QLabel):
    def __init__(self, text: str):
        super().__init__(text)
        font = QFont()
        font.setPointSize(14)
        font.setBold(True)
        self.setFont(font)
