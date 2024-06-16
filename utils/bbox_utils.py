from typing import List, Tuple

def get_center_of_bbox(bbox: List[float]) -> Tuple[int, int]:
    x1, y1, x2, y2 = bbox

    return int((x1 + x2) / 2), int((y1 + y2) / 2)

def get_bbox_dimensions(bbox: List[float]) -> Tuple[float]:
    return bbox[2] - bbox[0], bbox[3] - bbox[1]    # x2 - x1, y2 - y1