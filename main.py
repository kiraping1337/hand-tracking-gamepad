import cv2
import mediapipe as mp
import numpy as np
import math
import vgamepad as vg

#MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)

#утилита рисования
mp_draw = mp.solutions.drawing_utils
draw_spec = mp_draw.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=1)

#вебка
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

#геймпад
gamepad = vg.VX360Gamepad()

#цвета для пальцев
finger_colors = {
    "thumb": (0, 100, 100),   # yellow
    "index": (0, 0, 255),    # red
    "middle": (0, 120, 0),   # green
    "ring": (120, 0, 0),     # blue
    "pinky": (255, 0, 255)   # magenta
}

finger_landmarks = {
    "thumb": [1, 2, 3, 4],
    "index": [5, 6, 7, 8],
    "middle": [9, 10, 11, 12],
    "ring": [13, 14, 15, 16],
    "pinky": [17, 18, 19, 20]
}
#переменные для управления
dead_zone_radius = 15
background_mode = "camera"
mirror_mode = True
sensivity = 3

#метод, определяющий загнут ли палец (кроме большого)
def is_extended(lm, tip, pip):
    return lm[tip].y < lm[pip].y

#метод, определяющий загнут ли большой палец
def is_thumb_extended(lm, hand_label):
    tip_x = lm[4].x
    ip_x = lm[3].x

    if hand_label == "Right":
        return tip_x < ip_x
    else:
        return tip_x > ip_x

#распознавание пальцев
def recognize_fingers(hand_landmarks, hand_label):
    lm = hand_landmarks.landmark
    return {
        "thumb": is_thumb_extended(lm, hand_label),
        "index": is_extended(lm, 8, 6),
        "middle": is_extended(lm, 12, 10),
        "ring": is_extended(lm, 16, 14),
        "pinky": is_extended(lm, 20, 18)
    }
#основной цикл программы
while True:
    success, frame = cap.read()
    if not success:
        break

    if mirror_mode:
        frame = cv2.flip(frame, 1) #отзеркаливание

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if background_mode == "blue":
        img = np.full_like(frame, (255, 50, 50))   #фон синий
    elif background_mode == "green":
        img = np.full_like(frame, (50, 255, 50))   #фон зеленый
    else:
        img = frame.copy()  #обычная камера

    h, w, _ = img.shape
    center_left = (w//4, h//2+30)
    center_right = (3*w //4, h//2+30)

    cv2.circle(img, center_left, dead_zone_radius, (255, 255, 255), 2)
    cv2.circle(img, center_right, dead_zone_radius, (255, 255, 255), 2)

    left_stick_x, left_stick_y = 0.0, 0.0
    right_stick_x, right_stick_y = 0.0, 0.0
    btn_a = btn_lb = btn_rb = btn_start = False

    if results.multi_hand_landmarks and results.multi_handedness:
        for handLms, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            label = handedness.classification[0].label
            lm = handLms.landmark

            mp_draw.draw_landmarks(
                img, handLms, mp_hands.HAND_CONNECTIONS,
                landmark_drawing_spec=draw_spec,
                connection_drawing_spec=draw_spec
            )

            cx, cy = int(lm[9].x * w), int(lm[9].y * h)  #координаты центра ладони
            if cx < w // 2:
                target = center_left
                zone_name = "Left Zone"
            else:
                target = center_right
                zone_name = "Right Zone"

            #координаты центра целевой зоны
            tx, ty = target

            dx = cx - tx
            dy = cy - ty

            #нормализованное расстояние (для мертвой зоны)
            distance = math.hypot(dx, dy)
            if distance > dead_zone_radius:
                #определяем направление
                direction_x = "Right" if dx > 0 else "Left"
                direction_y = "Down" if dy > 0 else "Up"

                text_org = (w//4-10, h-10) if zone_name == "Left Zone" else (3*w//4-10,h-10)
                cv2.putText(img, (direction_x + " and "+ direction_y), text_org,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)
                if zone_name == "Left Zone":
                    cv2.line(img, center_left, (cx, cy), (255, 255, 255), 2)
                else:
                    cv2.line(img, center_right, (cx, cy), (255, 255, 255), 2)

                #нормализованные координаты
                norm_dx = max(-1.0, min(1.0, dx / (w / 4) * sensivity))
                norm_dy = -max(-1.0, min(1.0, dy / (h / 2) * sensivity))

                #логика управления стиками геймпада
                if zone_name == "Left Zone":
                    left_stick_x = norm_dx
                    left_stick_y = norm_dy

                else:
                    right_stick_x = norm_dx
                    right_stick_y = norm_dy

                gamepad.update()

            #рисование рук
            fingers = recognize_fingers(handLms, label)
            for fname, ids in finger_landmarks.items():
                col = finger_colors[fname]
                for i in range(len(ids) - 1):
                    x1, y1 = int(lm[ids[i]].x * w), int(lm[ids[i]].y * h)
                    x2, y2 = int(lm[ids[i+1]].x * w), int(lm[ids[i+1]].y * h)
                    cv2.line(img, (x1, y1), (x2, y2), col, 3)

            #подписи
            base_x = int(lm[0].x * w) + 40
            base_y = int(lm[0].y * h)
            for i, (fname, up) in enumerate(fingers.items()):
                status = "Up" if up else "Down"
                cv2.putText(img, f"{fname}:{status}", (base_x, base_y + i*20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, finger_colors[fname], 2)

            cv2.putText(img, f"{label} hand", (base_x, base_y - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            #логика управления кнопками геймпада
            if zone_name =="Left Zone":
                if not fingers["index"] and not fingers["middle"] and not fingers["ring"] and not fingers["pinky"]:
                    btn_a = True
                    cv2.putText(img, ("A pressed"), (w // 4 - 10, h // 5 + 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)

                if not fingers["middle"] and not fingers["ring"] and fingers["index"] and fingers["pinky"] and not fingers["thumb"]:
                    btn_lb = True
                    cv2.putText(img, ("LB pressed"), (w // 4 - 10, h // 5 + 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)

            else:
                if not fingers["index"] and not fingers["middle"] and not fingers["ring"] and not fingers["pinky"]:
                    btn_rb = True
                    cv2.putText(img, ("RB pressed"), (3*w // 4 - 10, h // 5 + 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)

                if not fingers["middle"] and not fingers["ring"] and fingers["index"] and fingers["pinky"] and not fingers["thumb"]:
                    btn_start = True
                    cv2.putText(img, ("MENU pressed"), (3*w // 4 - 10, h // 5 + 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)

    #применение данных к геймпаду
    gamepad.left_joystick_float(x_value_float=left_stick_x, y_value_float=left_stick_y)
    gamepad.right_joystick_float(x_value_float=right_stick_x, y_value_float=right_stick_y)

    if btn_a:
        gamepad.press_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
    else:
        gamepad.release_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_A)

    if btn_lb:
        gamepad.press_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_LEFT_SHOULDER)
    else:
        gamepad.release_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_LEFT_SHOULDER)

    if btn_rb:
        gamepad.press_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_RIGHT_SHOULDER)
    else:
        gamepad.release_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_RIGHT_SHOULDER)

    if btn_start:
        gamepad.press_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_START)
    else:
        gamepad.release_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_START)

    gamepad.update()

    cv2.imshow("Hand Tracking Gamepad", img)

    #смена режимов
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('b'):
        background_mode = "blue"
    elif key == ord('g'):
        background_mode = "green"
    elif key == ord('r'):
        background_mode = "camera"
    elif key == ord('m'):
        mirror_mode = not mirror_mode

cap.release()
cv2.destroyAllWindows()
