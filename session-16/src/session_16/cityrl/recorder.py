import cv2
import numpy as np
from PySide6.QtCore import QThread
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QWidget


class VideoSaver(QThread):
    def __init__(self, frames: list[np.ndarray], file_path: str, fps: int):
        super().__init__()

        self.frames = frames
        self.file_path = file_path
        self.fps = fps

    def run(self):
        height, width = self.frames[0].shape[:2]
        out = cv2.VideoWriter(
            self.file_path,
            cv2.VideoWriter.fourcc(*"mp4v"),
            self.fps,
            (width, height),
        )

        for frame in self.frames:
            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        out.release()


class Recorder:
    def __init__(self, widget: QWidget, fps: int, save_prefix: str):
        self.widget = widget
        self.frames: list[np.ndarray] = []
        self.count = 0
        self.prefix = save_prefix
        self.fps = fps

        self.active_savers: list[VideoSaver] = []

    def capture(self):
        pixmap = self.widget.grab()
        img = pixmap.toImage()
        img = img.convertToFormat(QImage.Format.Format_RGB888)

        width = img.width()
        height = img.height()
        ptr = img.constBits()

        total_bytes = img.sizeInBytes()
        arr = np.frombuffer(ptr, np.uint8, total_bytes)

        bytes_per_line = img.bytesPerLine()
        arr = arr.reshape((height, bytes_per_line))[:, : width * 3]
        arr = arr.reshape((height, width, 3))

        self.frames.append(arr.copy())

    def save(self):
        if len(self.frames) == 0:
            return

        file_saver = VideoSaver(
            self.frames, f"{self.prefix}_{self.count}.mp4", self.fps
        )
        file_saver.finished.connect(lambda: self._cleanup_saver(file_saver))
        file_saver.start()

        self.frames = []
        self.count += 1

    def clear(self):
        self.frames = []

    def reset(self):
        self.frames = []
        self.count = 0

    def _cleanup_saver(self, saver: VideoSaver):
        if saver in self.active_savers:
            self.active_savers.remove(saver)
