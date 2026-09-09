import time
from dataclasses import dataclass

import cv2

REQUEST_WIDTH = 1280
REQUEST_HEIGHT = 720
REQUEST_FPS = 60


@dataclass(frozen=True)
class CameraProfile:
    backend: int
    name: str
    use_mjpg: bool


_profiles: dict[int, CameraProfile] = {}


def _configure(cap: cv2.VideoCapture, use_mjpg: bool) -> None:
    if use_mjpg:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, REQUEST_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, REQUEST_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, REQUEST_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)


def _wait_frame(cap: cv2.VideoCapture, attempts: int = 40) -> bool:
    for _ in range(attempts):
        ok, frame = cap.read()
        if ok and frame is not None:
            return True
        time.sleep(0.05)
    return False


def _try_open(index: int, backend: int, name: str, use_mjpg: bool):
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        cap.release()
        return None
    _configure(cap, use_mjpg)
    if not _wait_frame(cap):
        cap.release()
        return None
    print(
        f"[camera] {name}{'+MJPG' if use_mjpg else ''} "
        f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
        f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} -> принято"
    )
    return cap


def open_capture(index: int) -> cv2.VideoCapture | None:
    cached = _profiles.get(index)
    attempts: list[tuple[int, str, bool]] = []
    if cached is not None:
        attempts.append((cached.backend, cached.name, cached.use_mjpg))
    attempts.extend(
        [
            (cv2.CAP_MSMF, "MSMF", True),
            (cv2.CAP_MSMF, "MSMF", False),
            (cv2.CAP_ANY, "ANY", False),
        ]
    )

    seen: set[tuple[int, bool]] = set()
    for backend, name, mjpg in attempts:
        key = (backend, mjpg)
        if key in seen:
            continue
        seen.add(key)
        cap = _try_open(index, backend, name, mjpg)
        if cap is not None:
            _profiles[index] = CameraProfile(backend, name, mjpg)
            return cap
    print(f"[camera] не удалось открыть индекс {index}")
    return None


def invalidate_profile(index: int) -> None:
    _profiles.pop(index, None)
