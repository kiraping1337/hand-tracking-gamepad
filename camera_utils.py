#возвращает список доступных камер
def list_cameras(max_index: int = 2, skip: set[int] | None = None) -> list[int]:
    skip = skip or set()
    cameras = list(range(max_index))
    for index in skip:
        if index not in cameras:
            cameras.insert(0, index)
    return cameras
