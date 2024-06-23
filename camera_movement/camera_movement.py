from typing import List, Dict
import cv2
import numpy as np
import time
from datetime import datetime
import logging
from utils import get_distance, options

file_handler = logging.FileHandler("logs/camera_movement.log")
file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

logger = logging.getLogger("camera_movement")
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.addHandler(stream_handler)

class CameraMovementEstimator():
    def __init__(self, frame: np.ndarray, classes: List[int], verbose: bool=True) -> None:
        self.minimum_distance = 5       # minimum the camera needs to move

        # lucas kanade optical flow
        self.lk_params = dict(
            winSize = (15, 15),
            maxLevel = 2,       # number of pyramid layers
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )

        first_frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # take top and bottom as orientation for the movement detection as there is not much change in the movement
        mask_features = np.zeros_like(first_frame_grayscale)
        mask_features[:, 0:20] = 1
        mask_features[:, frame.shape[0]-150:frame.shape[0]] = 1 

        self.features = dict(
            maxCorners = 100,       # maximum number of corners (features) to be detected
            qualityLevel = 0.3,     # only corners above this level
            minDistance = 10,       # ensures that corners are not too close to each other --> better distribution of the corners
            blockSize = 7,          # window size for corner detection
            mask = mask_features    # area of the image
        )

        self.classes = classes
        self.verbose = verbose

    def adjust_positions_to_tracks(self, tracks: Dict[str, List[Dict]], camera_movement_per_frame: List[List[float]]) -> None:
        for object, object_tracks in tracks.items():
            for frame_num, track_dict in enumerate(object_tracks):
                for tracker_id, track in track_dict.items():
                    position = track["position"]
                    camera_movement = camera_movement_per_frame[frame_num]
                    position_adjusted = (position[0] - camera_movement[0], position[1] - camera_movement[1])    # x - camera_x, y - camera_y
                    
                    tracks[object][frame_num][tracker_id]["position_adjusted"] = position_adjusted

    def get_camera_movement(self, frames: List[np.ndarray]) -> List[List[float]]:
        start_time = time.time()

        if self.verbose:
            logger.info(f"Starting camera movement detection at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        camera_movement = [[0, 0]] * len(frames) # x, y

        old_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)  # convert image to gray
        old_features = cv2.goodFeaturesToTrack(old_gray, **self.features)   # ** to expand dictionary into the parameters

        for frame_num in range(1, len(frames)):
            frame_gray = cv2.cvtColor(frames[frame_num], cv2.COLOR_BGR2GRAY)
            new_features, _, _ = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, old_features, None, **self.lk_params)

            max_distance = 0
            camera_movement_x, camera_movement_y = 0, 0

            for i, (new, old) in enumerate(zip(new_features, old_features)):
                new_features_point = new.ravel()
                old_features_point = old.ravel()

                distance = get_distance(new_features_point, old_features_point)

                if distance > max_distance:
                    max_distance = distance
                    camera_movement_x = new_features_point[0] - old_features_point[0]
                    camera_movement_y = new_features_point[1] - old_features_point[1]

            if max_distance > self.minimum_distance:
                camera_movement[frame_num] = [camera_movement_x, camera_movement_y]
                old_features = cv2.goodFeaturesToTrack(frame_gray, **self.features)

            old_gray = frame_gray.copy()

        if self.verbose:
            logger.info(f"Processed camera movement in {len(frames)} frames in {time.time() - start_time:.2f} seconds.")
            
            separator = f"{'-'*10} [End of camera movement detection] at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {'-'*10}"
            logger.info(separator)

        return camera_movement
    
    def draw_camera_movement(self, frames: List[np.ndarray], camera_movement_per_frame: List[List[float]]) -> List[np.ndarray]:
        output_frames = []

        if options["stats"] in self.classes:
            for frame_num, frame in enumerate(frames):
                frame = frame.copy()

                overlay = frame.copy()

                cv2.rectangle(overlay, pt1=(0, 0), pt2=(500, 100), color=(255, 255, 255), thickness=cv2.FILLED)
                alpha = 0.4
                cv2.addWeighted(src1=overlay, alpha=alpha, src2=frame, beta=1-alpha, gamma=0, dst=frame) 

                x_movement, y_movement = camera_movement_per_frame[frame_num]
    
                frame = cv2.putText(frame, text=f"Camera Movement X: {x_movement:.2f}", org=(10, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 0), thickness=3)
                frame = cv2.putText(frame, text=f"Camera Movement Y: {y_movement:.2f}", org=(10, 60), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 0), thickness=3)

                output_frames.append(frame)
        else:
            output_frames = frames.copy()

        return output_frames