import time

import cv2
import numpy as np
from PyQt6.QtCore import QMutex, QThread, pyqtSignal
from PyQt6.QtGui import QImage

from camera_open import invalidate_profile, open_capture
from config import TrackingConfig
from hand_tracker import HandTracker

OPEN_RETRIES = 3
OPEN_RETRY_DELAY_S = 0.4


def _frame_to_qimage(frame_bgr: np.ndarray) -> QImage:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rgb = np.ascontiguousarray(rgb)
    h, w, ch = rgb.shape
    return QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()


class _FrameGrabber(QThread):
    def __init__(self, camera_index: int):
        super().__init__()
        self._camera_index = camera_index
        self._mutex = QMutex()
        self._frame: np.ndarray | None = None
        self._running = False
        self._open_ok = False

    @property
    def open_ok(self) -> bool:
        return self._open_ok

    def take_latest(self) -> np.ndarray | None:
        self._mutex.lock()
        frame = self._frame
        self._frame = None
        self._mutex.unlock()
        return frame

    def run(self) -> None:
        self._running = True
        cap = None
        for attempt in range(OPEN_RETRIES):
            if not self._running:
                break
            if attempt > 0:
                invalidate_profile(self._camera_index)
                time.sleep(OPEN_RETRY_DELAY_S * attempt)
            cap = open_capture(self._camera_index)
            if cap is not None:
                break

        if cap is None:
            print(
                f"[camera] grabber: не удалось открыть камеру {self._camera_index}"
            )
            return

        self._open_ok = True
        while self._running:
            success, frame = cap.read()
            if not success or frame is None:
                self.msleep(5)
                continue
            self._mutex.lock()
            self._frame = frame
            self._mutex.unlock()
        cap.release()

    def stop(self) -> None:
        self._running = False
        self.wait(8000)


class CameraWorker(QThread):
    frame_ready = pyqtSignal(QImage)
    fps_ready = pyqtSignal(float)
    source_size_ready = pyqtSignal(int, int)
    status_changed = pyqtSignal(str)

    def __init__(
        self,
        config: TrackingConfig,
        tracker: HandTracker,
        camera_index: int = 0,
    ):
        super().__init__()
        self.config = config
        self.tracker = tracker
        self._mutex = QMutex()
        self._running = False
        self._camera_index = camera_index
        self._restart_camera = camera_index >= 0
        self._display_width = 960
        self._last_source_size = (0, 0)

    def set_camera(self, index: int, *, force: bool = False) -> None:
        self._mutex.lock()
        if index == self._camera_index and not force:
            self._mutex.unlock()
            return
        self._camera_index = index
        self._restart_camera = True
        self._mutex.unlock()

    def set_display_width(self, width: int) -> None:
        width = max(160, int(width))
        self._mutex.lock()
        self._display_width = width
        self._mutex.unlock()

    def stop(self) -> None:
        self._running = False
        self.wait(8000)

    def run(self) -> None:
        self._running = True
        grabber: _FrameGrabber | None = None
        frames = 0
        t0 = time.perf_counter()

        while self._running:
            self._mutex.lock()
            restart = self._restart_camera
            index = self._camera_index
            display_width = self._display_width
            if restart:
                self._restart_camera = False
            self._mutex.unlock()

            if restart:
                if grabber is not None:
                    grabber.stop()
                    grabber = None
                if index >= 0:
                    self.status_changed.emit(
                        "Подождите немного, подключаем камеру…"
                    )
                    grabber = _FrameGrabber(index)
                    grabber.start()

            if grabber is None:
                self.msleep(50)
                continue

            if not grabber.open_ok:
                if grabber.isFinished():
                    grabber = None
                    self._mutex.lock()
                    self._restart_camera = True
                    self._mutex.unlock()
                    self.msleep(400)
                else:
                    self.msleep(20)
                continue

            frame = grabber.take_latest()
            if frame is None:
                self.msleep(2)
                continue

            if not self.tracker.is_ready:
                self._emit_frame(self.tracker.process_frame(frame), display_width)
                self.status_changed.emit("Камера подключена, загружаем трекинг…")
                self.tracker.ensure_ready()
                self.status_changed.emit("")
                continue

            self._emit_frame(self.tracker.process_frame(frame), display_width)

            frames += 1
            now = time.perf_counter()
            if now - t0 >= 1.0:
                self.fps_ready.emit(frames / (now - t0))
                frames = 0
                t0 = now

        if grabber is not None:
            grabber.stop()

    def _emit_frame(self, processed, display_width: int) -> None:
        src_size = (processed.shape[1], processed.shape[0])
        if src_size != self._last_source_size:
            self._last_source_size = src_size
            self.source_size_ready.emit(src_size[0], src_size[1])

        out = processed
        src_w = out.shape[1]
        if display_width < src_w:
            k = display_width / src_w
            out = cv2.resize(
                out,
                (display_width, max(1, int(round(out.shape[0] * k)))),
                interpolation=cv2.INTER_LINEAR,
            )
        self.frame_ready.emit(_frame_to_qimage(out))
