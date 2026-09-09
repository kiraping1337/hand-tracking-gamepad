import math

import cv2
import mediapipe as mp
import numpy as np
import vgamepad as vg

from config import TrackingConfig

#MediaPipe
mp_hands = mp.solutions.hands

#утилита рисования
mp_draw = mp.solutions.drawing_utils
DRAW_SPEC = mp_draw.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=1)

#цвета для пальцев
FINGER_COLORS = {
    "thumb": (0, 100, 100),
    "index": (0, 0, 255),
    "middle": (0, 120, 0),
    "ring": (120, 0, 0),
    "pinky": (255, 0, 255),
}

FINGER_LANDMARKS = {
    "thumb": [1, 2, 3, 4],
    "index": [5, 6, 7, 8],
    "middle": [9, 10, 11, 12],
    "ring": [13, 14, 15, 16],
    "pinky": [17, 18, 19, 20],
}

DEFAULT_DETECT_WIDTH = 0

#метод, определяющий загнут ли палец (кроме большого)
def is_extended(lm, tip: int, pip: int) -> bool:
    return lm[tip].y < lm[pip].y

#метод, определяющий загнут ли большой палец
def is_thumb_extended(lm, hand_label: str) -> bool:
    tip_x = lm[4].x
    ip_x = lm[3].x
    if hand_label == "Right":
        return tip_x < ip_x
    return tip_x > ip_x

#распознавание пальцев
def recognize_fingers(hand_landmarks, hand_label: str) -> dict[str, bool]:
    lm = hand_landmarks.landmark
    return {
        "thumb": is_thumb_extended(lm, hand_label),
        "index": is_extended(lm, 8, 6),
        "middle": is_extended(lm, 12, 10),
        "ring": is_extended(lm, 16, 14),
        "pinky": is_extended(lm, 20, 18),
    }


class HandTracker:
    def __init__(self, config: TrackingConfig | None = None):
        self.config = config or TrackingConfig()
        self.hands = None
        self.gamepad = None
        self._canvas: np.ndarray | None = None
        self._small: np.ndarray | None = None

    @property
    def is_ready(self) -> bool:
        return self.hands is not None

    def ensure_ready(self) -> None:
        if self.hands is None:
            self.hands = mp_hands.Hands(
                max_num_hands=2,
                model_complexity=0,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.5,
            )
        if self.gamepad is None:
            self.gamepad = vg.VX360Gamepad()

    def close(self) -> None:
        if self.hands is not None:
            self.hands.close()
            self.hands = None
        if self.gamepad is not None:
            self.gamepad.reset()
            self.gamepad.update()

    #вспомогательное

    def _solid_canvas(self, shape, color) -> np.ndarray:
        if self._canvas is None or self._canvas.shape != shape:
            self._canvas = np.empty(shape, dtype=np.uint8)
        self._canvas[:] = color
        return self._canvas

    def _detect(self, frame_bgr: np.ndarray):
        h, w = frame_bgr.shape[:2]
        target_w = int(getattr(self.config, "detect_width", DEFAULT_DETECT_WIDTH))

        if 0 < target_w < w:
            target_h = max(1, int(round(h * target_w / w)))
            small = cv2.resize(
                frame_bgr, (target_w, target_h), interpolation=cv2.INTER_AREA
            )
        else:
            small = frame_bgr

        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        return self.hands.process(rgb)

    #основной кадр

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        cfg = self.config

        if cfg.mirror_mode:
            frame = cv2.flip(frame, 1) #отзеркаливание


        if self.hands is None:
            img = frame if cfg.mirror_mode else frame.copy()
            h, w, _ = img.shape
            cfg.ensure_centers(w, h)
            cv2.circle(img, cfg.center_left, cfg.dead_zone_radius, (255, 255, 255), 2)
            cv2.circle(img, cfg.center_right, cfg.dead_zone_radius, (255, 255, 255), 2)
            return img

        results = self._detect(frame)

        if cfg.background_mode == "blue":
            img = self._solid_canvas(frame.shape, (255, 50, 50)) #фон синий
        elif cfg.background_mode == "green":
            img = self._solid_canvas(frame.shape, (50, 255, 50)) #фон зеленый
        else:
            img = frame #обычная камера

        h, w, _ = img.shape
        cfg.ensure_centers(w, h)

        center_left = cfg.center_left
        center_right = cfg.center_right
        dead_zone_radius = cfg.dead_zone_radius

        cv2.circle(img, center_left, dead_zone_radius, (255, 255, 255), 2)
        cv2.circle(img, center_right, dead_zone_radius, (255, 255, 255), 2)

        left_stick_x, left_stick_y = 0.0, 0.0
        right_stick_x, right_stick_y = 0.0, 0.0
        btn_a = btn_lb = btn_rb = btn_start = False

        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_lms, handedness in zip(
                results.multi_hand_landmarks, results.multi_handedness
            ):
                label = handedness.classification[0].label
                lm = hand_lms.landmark

                mp_draw.draw_landmarks(
                    img,
                    hand_lms,
                    mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=DRAW_SPEC,
                    connection_drawing_spec=DRAW_SPEC,
                )

                cx, cy = int(lm[9].x * w), int(lm[9].y * h) #координаты центра ладони
                if cx < w // 2:
                    target = center_left
                    zone_name = "Left Zone"
                else:
                    target = center_right
                    zone_name = "Right Zone"

                tx, ty = target
                dx = cx - tx
                dy = cy - ty
                distance = math.hypot(dx, dy) #нормализованное расстояние (для мертвой зоны)

                if distance > dead_zone_radius:
                    #определяем направление
                    direction_x = "Right" if dx > 0 else "Left"
                    direction_y = "Down" if dy > 0 else "Up"

                    text_org = (
                        (w // 4 - 10, h - 10)
                        if zone_name == "Left Zone"
                        else (3 * w // 4 - 10, h - 10)
                    )
                    cv2.putText(
                        img,
                        direction_x + " and " + direction_y,
                        text_org,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (255, 255, 0),
                        2,
                    )
                    if zone_name == "Left Zone":
                        cv2.line(img, center_left, (cx, cy), (255, 255, 255), 2)
                    else:
                        cv2.line(img, center_right, (cx, cy), (255, 255, 255), 2)

                    #нормализованные координаты
                    norm_dx = max(-1.0, min(1.0, dx / (w / 4) * cfg.sensitivity))
                    norm_dy = -max(-1.0, min(1.0, dy / (h / 2) * cfg.sensitivity))

                    #логика управления стиками геймпада
                    if zone_name == "Left Zone":
                        left_stick_x = norm_dx
                        left_stick_y = norm_dy
                    else:
                        right_stick_x = norm_dx
                        right_stick_y = norm_dy

                #рисование рук
                fingers = recognize_fingers(hand_lms, label)
                for fname, ids in FINGER_LANDMARKS.items():
                    col = FINGER_COLORS[fname]
                    for i in range(len(ids) - 1):
                        x1, y1 = int(lm[ids[i]].x * w), int(lm[ids[i]].y * h)
                        x2, y2 = int(lm[ids[i + 1]].x * w), int(lm[ids[i + 1]].y * h)
                        cv2.line(img, (x1, y1), (x2, y2), col, 3)

                #подписи
                base_x = int(lm[0].x * w) + 40
                base_y = int(lm[0].y * h)
                for i, (fname, up) in enumerate(fingers.items()):
                    status = "Up" if up else "Down"
                    cv2.putText(
                        img,
                        f"{fname}:{status}",
                        (base_x, base_y + i * 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        FINGER_COLORS[fname],
                        2,
                    )

                cv2.putText(
                    img,
                    f"{label} hand",
                    (base_x, base_y - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                )
                #логика управления кнопками геймпада
                if zone_name == "Left Zone":
                    if (
                        not fingers["index"]
                        and not fingers["middle"]
                        and not fingers["ring"]
                        and not fingers["pinky"]
                    ):
                        btn_a = True
                        cv2.putText(
                            img,
                            "A pressed",
                            (w // 4 - 10, h // 5 + 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.65,
                            (255, 255, 0),
                            2,
                        )

                    if (
                        not fingers["middle"]
                        and not fingers["ring"]
                        and fingers["index"]
                        and fingers["pinky"]
                        and not fingers["thumb"]
                    ):
                        btn_lb = True
                        cv2.putText(
                            img,
                            "LB pressed",
                            (w // 4 - 10, h // 5 + 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.65,
                            (255, 255, 0),
                            2,
                        )
                else:
                    if (
                        not fingers["index"]
                        and not fingers["middle"]
                        and not fingers["ring"]
                        and not fingers["pinky"]
                    ):
                        btn_rb = True
                        cv2.putText(
                            img,
                            "RB pressed",
                            (3 * w // 4 - 10, h // 5 + 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.65,
                            (255, 255, 0),
                            2,
                        )

                    if (
                        not fingers["middle"]
                        and not fingers["ring"]
                        and fingers["index"]
                        and fingers["pinky"]
                        and not fingers["thumb"]
                    ):
                        btn_start = True
                        cv2.putText(
                            img,
                            "MENU pressed",
                            (3 * w // 4 - 10, h // 5 + 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.65,
                            (255, 255, 0),
                            2,
                        )
        #применение данных к геймпаду
        self.gamepad.left_joystick_float(
            x_value_float=left_stick_x, y_value_float=left_stick_y
        )
        self.gamepad.right_joystick_float(
            x_value_float=right_stick_x, y_value_float=right_stick_y
        )

        if btn_a:
            self.gamepad.press_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
        else:
            self.gamepad.release_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_A)

        if btn_lb:
            self.gamepad.press_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_LEFT_SHOULDER)
        else:
            self.gamepad.release_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_LEFT_SHOULDER)

        if btn_rb:
            self.gamepad.press_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_RIGHT_SHOULDER)
        else:
            self.gamepad.release_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_RIGHT_SHOULDER)

        if btn_start:
            self.gamepad.press_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_START)
        else:
            self.gamepad.release_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_START)

        self.gamepad.update()
        return img