from typing import List, Tuple
import math

def get_center_of_bbox(bbox: List[float]) -> Tuple[int, int]:
    x1, y1, x2, y2 = bbox

    return int((x1 + x2) / 2), int((y1 + y2) / 2)

def get_bbox_dimensions(bbox: List[float]) -> Tuple[int, int]:
    return int(bbox[2] - bbox[0]), int(bbox[3] - bbox[1])    # x2 - x1, y2 - y1

def get_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return  math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)  # eucledian distance of points p1 and p2

def get_foot_position(bbox: List[float]) -> Tuple[int, int]:    # center of x, bottom of y
    x1, _, x2, y2 = bbox

    return int((x1 + x2) / 2), int(y2)