from PyQt6.QtCore import QRect, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QPainter, QPixmap
from PyQt6.QtWidgets import QWidget

BACKGROUND = QColor("#1a1a1a")
PLACEHOLDER = QColor("#e8e8e8")


class VideoLabel(QWidget):

    size_changed = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, True)
        self._source: QPixmap | None = None
        self._dest_rect = QRect()
        self._placeholder = "Подождите немного…\nИдёт подключение камеры"

    def set_placeholder(self, text: str) -> None:
        self._placeholder = text
        if self._source is None:
            self.update()

    def set_frame(self, pixmap: QPixmap) -> None:
        self._source = pixmap
        self._update_dest_rect()
        self.update()

    def clear_frame(self) -> None:
        self._source = None
        self._dest_rect = QRect()
        self.update()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._update_dest_rect()
        self.size_changed.emit(int(self.width() * self.devicePixelRatioF()))

    def _update_dest_rect(self) -> None:
        if self._source is None or self._source.isNull():
            self._dest_rect = QRect()
            return

        src_w = self._source.width()
        src_h = self._source.height()
        if src_w <= 0 or src_h <= 0 or self.width() <= 0 or self.height() <= 0:
            self._dest_rect = QRect()
            return

        scale = min(self.width() / src_w, self.height() / src_h)
        dest_w = int(src_w * scale)
        dest_h = int(src_h * scale)
        dest_x = (self.width() - dest_w) // 2
        dest_y = (self.height() - dest_h) // 2
        self._dest_rect = QRect(dest_x, dest_y, dest_w, dest_h)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)

        if self._source is None or self._source.isNull() or self._dest_rect.isEmpty():
            painter.fillRect(self.rect(), BACKGROUND)
            painter.setPen(PLACEHOLDER)
            font = QFont(self.font())
            font.setPointSize(14)
            painter.setFont(font)
            painter.drawText(
                self.rect().adjusted(24, 24, -24, -24),
                Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap,
                self._placeholder + "\n\nЭто может занять до минуты",
            )
            return

        region_top = QRect(0, 0, self.width(), self._dest_rect.top())
        region_bottom = QRect(
            0,
            self._dest_rect.bottom() + 1,
            self.width(),
            self.height() - self._dest_rect.bottom() - 1,
        )
        region_left = QRect(0, 0, self._dest_rect.left(), self.height())
        region_right = QRect(
            self._dest_rect.right() + 1,
            0,
            self.width() - self._dest_rect.right() - 1,
            self.height(),
        )
        for r in (region_top, region_bottom, region_left, region_right):
            if r.isValid() and not r.isEmpty():
                painter.fillRect(r, BACKGROUND)

        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
        painter.drawPixmap(self._dest_rect, self._source)
