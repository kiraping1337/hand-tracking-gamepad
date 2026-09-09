from dataclasses import dataclass


@dataclass
class TrackingConfig:
    dead_zone_radius: int = 15
    center_left_x: int | None = None
    center_left_y: int | None = None
    center_right_x: int | None = None
    center_right_y: int | None = None
    mirror_mode: bool = True
    background_mode: str = "camera"
    sensitivity: float = 3.0
    detect_width: int = 0  # 0 = полный кадр для MediaPipe

    def ensure_centers(self, width: int, height: int) -> None:
        if self.center_left_x is None:
            self.center_left_x = width // 4
        if self.center_left_y is None:
            self.center_left_y = height // 2 + 30
        if self.center_right_x is None:
            self.center_right_x = 3 * width // 4
        if self.center_right_y is None:
            self.center_right_y = height // 2 + 30

    @property
    def center_left(self) -> tuple[int, int]:
        return self.center_left_x or 0, self.center_left_y or 0

    @property
    def center_right(self) -> tuple[int, int]:
        return self.center_right_x or 0, self.center_right_y or 0

    def set_center_left(self, x: int, y: int) -> None:
        self.center_left_x = x
        self.center_left_y = y

    def set_center_right(self, x: int, y: int) -> None:
        self.center_right_x = x
        self.center_right_y = y
