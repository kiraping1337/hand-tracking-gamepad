from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QImage, QKeyEvent, QPixmap
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from config import TrackingConfig
from hand_tracker import HandTracker
from ui.camera_worker import CameraWorker
from ui.video_label import VideoLabel


class PreviewWindow(QWidget):
    closed = pyqtSignal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent, Qt.WindowType.Window)
        self.setWindowTitle("Камера — отдельное окно")
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self.resize(960, 540)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.video_label = VideoLabel()
        self.video_label.setMinimumSize(640, 360)
        layout.addWidget(self.video_label)

    def update_frame(self, pixmap: QPixmap) -> None:
        self.video_label.set_frame(pixmap)

    def closeEvent(self, event) -> None:
        self.closed.emit()
        super().closeEvent(event)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Hand Tracking Gamepad")
        self.resize(1280, 720)

        self.config = TrackingConfig()
        self.tracker = HandTracker(self.config)
        self.preview_window: PreviewWindow | None = None
        self._updating_spinboxes = False
        self._frame_size_initialized = False
        self._frame_width = 1280
        self._frame_height = 720
        self._source_pixmap: QPixmap | None = None
        self._cameras_cache: list[int] | None = None

        self._build_ui()

        self.worker = CameraWorker(self.config, self.tracker, camera_index=0)
        self.worker.frame_ready.connect(self._on_frame_ready)
        self.worker.fps_ready.connect(self._on_fps)
        self.worker.source_size_ready.connect(self._on_source_size)
        self.worker.status_changed.connect(self._on_status)
        self.worker.start()

        self.video_label.size_changed.connect(self._on_display_size_changed)

        self._populate_camera_combo([0, 1], select_index=0)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._on_display_size_changed(0)

    #UI

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QHBoxLayout(central)

        settings_panel = QWidget()
        settings_panel.setFixedWidth(280)
        settings_layout = QVBoxLayout(settings_panel)

        camera_group = QGroupBox("Камера")
        camera_layout = QVBoxLayout(camera_group)

        camera_row = QHBoxLayout()
        self.camera_combo = QComboBox()
        self.camera_combo.currentIndexChanged.connect(self._on_camera_changed)
        camera_row.addWidget(self.camera_combo, stretch=1)

        refresh_btn = QPushButton("↻")
        refresh_btn.setFixedWidth(36)
        refresh_btn.setToolTip("Обновить список камер")
        refresh_btn.clicked.connect(self._rescan_cameras)
        camera_row.addWidget(refresh_btn)
        camera_layout.addLayout(camera_row)

        self.open_window_btn = QPushButton("Открыть камеру в отдельном окне")
        self.open_window_btn.clicked.connect(self._toggle_preview_window)
        camera_layout.addWidget(self.open_window_btn)

        self.fps_label = QLabel("Подождите немного, подключаем камеру…")
        self.fps_label.setWordWrap(True)
        camera_layout.addWidget(self.fps_label)

        settings_layout.addWidget(camera_group)

        zones_group = QGroupBox("Зоны управления")
        zones_form = QFormLayout(zones_group)

        self.dead_zone_spin = QSpinBox()
        self.dead_zone_spin.setRange(1, 500)
        self.dead_zone_spin.setValue(self.config.dead_zone_radius)
        self.dead_zone_spin.valueChanged.connect(self._on_dead_zone_changed)
        zones_form.addRow("Dead zone radius:", self.dead_zone_spin)

        self.left_x_spin = QSpinBox()
        self.left_x_spin.setRange(0, 4096)
        self.left_x_spin.valueChanged.connect(self._on_center_changed)
        zones_form.addRow("Центр левой зоны X:", self.left_x_spin)

        self.left_y_spin = QSpinBox()
        self.left_y_spin.setRange(0, 4096)
        self.left_y_spin.valueChanged.connect(self._on_center_changed)
        zones_form.addRow("Центр левой зоны Y:", self.left_y_spin)

        self.right_x_spin = QSpinBox()
        self.right_x_spin.setRange(0, 4096)
        self.right_x_spin.valueChanged.connect(self._on_center_changed)
        zones_form.addRow("Центр правой зоны X:", self.right_x_spin)

        self.right_y_spin = QSpinBox()
        self.right_y_spin.setRange(0, 4096)
        self.right_y_spin.valueChanged.connect(self._on_center_changed)
        zones_form.addRow("Центр правой зоны Y:", self.right_y_spin)

        reset_centers_btn = QPushButton("Сбросить центры по умолчанию")
        reset_centers_btn.clicked.connect(self._reset_centers)
        zones_form.addRow(reset_centers_btn)

        settings_layout.addWidget(zones_group)

        options_group = QGroupBox("Дополнительно")
        options_layout = QVBoxLayout(options_group)

        self.mirror_checkbox = QCheckBox("Зеркальный режим")
        self.mirror_checkbox.setChecked(self.config.mirror_mode)
        self.mirror_checkbox.toggled.connect(self._on_mirror_changed)
        options_layout.addWidget(self.mirror_checkbox)

        detect_form = QFormLayout()
        self.detect_width_spin = QSpinBox()
        self.detect_width_spin.setRange(0, 1280)  # 0 = полный кадр
        self.detect_width_spin.setSingleStep(32)
        self.detect_width_spin.setValue(self.config.detect_width)
        self.detect_width_spin.setToolTip(
            "Ширина кадра для MediaPipe. 0 = полный кадр.\n"
            "На скорость почти не влияет: MediaPipe масштабирует вход сам."
        )
        self.detect_width_spin.valueChanged.connect(self._on_detect_width_changed)
        detect_form.addRow("Разрешение трекинга:", self.detect_width_spin)
        options_layout.addLayout(detect_form)

        bg_label = QLabel("Фон: B — синий, G — зелёный, R — камера")
        bg_label.setWordWrap(True)
        options_layout.addWidget(bg_label)

        settings_layout.addWidget(options_group)
        settings_layout.addStretch()

        root_layout.addWidget(settings_panel)

        self.video_label = VideoLabel()
        self.video_label.setMinimumSize(640, 360)
        root_layout.addWidget(self.video_label, stretch=1)

    #камеры

    def _rescan_cameras(self) -> None:
        current = self.camera_combo.currentData()
        if current is None or current < 0:
            current = 0
        self._populate_camera_combo([0, 1], select_index=current)

    def _populate_camera_combo(
        self, cameras: list[int], *, select_index: int
    ) -> None:
        self.camera_combo.blockSignals(True)
        self.camera_combo.clear()

        if not cameras:
            self.camera_combo.addItem("Камеры не найдены", -1)
        else:
            for index in cameras:
                self.camera_combo.addItem(f"Камера {index}", index)
            idx = self.camera_combo.findData(select_index)
            if idx >= 0:
                self.camera_combo.setCurrentIndex(idx)

        self.camera_combo.blockSignals(False)

    def _on_camera_changed(self) -> None:
        camera_index = self.camera_combo.currentData()
        if camera_index is None or camera_index < 0:
            return
        self._frame_size_initialized = False
        self._source_pixmap = None
        self.video_label.clear_frame()
        self.video_label.set_placeholder(
            "Подождите немного…\nИдёт подключение камеры"
        )
        self.fps_label.setText("Подождите немного, подключаем камеру…")
        self.worker.set_camera(camera_index)

    #настройки

    def _update_spinbox_limits(self, width: int, height: int) -> None:
        for spin in (self.left_x_spin, self.right_x_spin):
            spin.setMaximum(width)
        for spin in (self.left_y_spin, self.right_y_spin):
            spin.setMaximum(height)

    def _sync_spinboxes_from_config(self) -> None:
        self._updating_spinboxes = True
        self.dead_zone_spin.setValue(self.config.dead_zone_radius)
        self.left_x_spin.setValue(self.config.center_left_x or 0)
        self.left_y_spin.setValue(self.config.center_left_y or 0)
        self.right_x_spin.setValue(self.config.center_right_x or 0)
        self.right_y_spin.setValue(self.config.center_right_y or 0)
        self._updating_spinboxes = False

    def _reset_centers(self) -> None:
        self.config.center_left_x = None
        self.config.center_left_y = None
        self.config.center_right_x = None
        self.config.center_right_y = None
        self.config.ensure_centers(self._frame_width, self._frame_height)
        self._sync_spinboxes_from_config()

    def _on_dead_zone_changed(self, value: int) -> None:
        self.config.dead_zone_radius = value

    def _on_detect_width_changed(self, value: int) -> None:
        self.config.detect_width = value

    def _on_center_changed(self) -> None:
        if self._updating_spinboxes:
            return
        self.config.set_center_left(self.left_x_spin.value(), self.left_y_spin.value())
        self.config.set_center_right(
            self.right_x_spin.value(), self.right_y_spin.value()
        )

    def _on_mirror_changed(self, checked: bool) -> None:
        self.config.mirror_mode = checked

    #превью

    def _toggle_preview_window(self) -> None:
        if self.preview_window is not None:
            self.preview_window.close()
            return

        self.preview_window = PreviewWindow()
        self.preview_window.closed.connect(self._on_preview_closed)
        self.preview_window.video_label.size_changed.connect(
            self._on_display_size_changed
        )
        self.preview_window.show()
        self.open_window_btn.setText("Закрыть отдельное окно")

        if self._source_pixmap is not None:
            self.preview_window.update_frame(self._source_pixmap)

    def _on_preview_closed(self) -> None:
        self.preview_window = None
        self.open_window_btn.setText("Открыть камеру в отдельном окне")
        self._on_display_size_changed(0)

    def _on_display_size_changed(self, _width: int) -> None:
        widths = [int(self.video_label.width() * self.devicePixelRatioF())]
        if self.preview_window is not None:
            widths.append(
                int(
                    self.preview_window.video_label.width()
                    * self.preview_window.devicePixelRatioF()
                )
            )
        self.worker.set_display_width(max(widths))

    #кадры
    def _on_source_size(self, width: int, height: int) -> None:
        self._frame_width = width
        self._frame_height = height
        self._update_spinbox_limits(width, height)
        self.config.ensure_centers(width, height)
        self._sync_spinboxes_from_config()
        self._frame_size_initialized = True

    def _on_frame_ready(self, image: QImage) -> None:
        self._source_pixmap = QPixmap.fromImage(image)
        self.video_label.set_frame(self._source_pixmap)

        if self.preview_window is not None:
            self.preview_window.update_frame(self._source_pixmap)

    def _on_fps(self, fps: float) -> None:
        self.fps_label.setText(f"FPS: {fps:.1f}")

    def _on_status(self, text: str) -> None:
        if text:
            self.fps_label.setText(text)
            self.video_label.set_placeholder(text.replace(" — ", "\n"))

    def keyPressEvent(self, event: QKeyEvent) -> None:
        key = event.key()
        if key == Qt.Key.Key_B:
            self.config.background_mode = "blue"
        elif key == Qt.Key.Key_G:
            self.config.background_mode = "green"
        elif key == Qt.Key.Key_R:
            self.config.background_mode = "camera"
        elif key == Qt.Key.Key_M:
            self.mirror_checkbox.setChecked(not self.config.mirror_mode)
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event) -> None:
        self.worker.stop()
        self.tracker.close()
        if self.preview_window is not None:
            self.preview_window.close()
        super().closeEvent(event)