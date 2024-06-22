import logging
from typing import List, Dict
import time
from datetime import datetime
import numpy as np
import pandas as pd
import ultralytics
import supervision as sv
from utils import ellipse, triangle, ball_possession_box, get_device, get_center_of_bbox, get_foot_position

handler = logging.FileHandler("logs/tracking.log")
handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger = logging.getLogger("tracker")
logger.setLevel(logging.INFO)
logger.addHandler(handler)

class Tracker:
    """
    Byte tracker. 
    Tracking persons by close bounding box in next frame combined with movement and visual features like shirt colour.
    Assigning bounding boxes unique IDs.
    Predicting and then tracking with supervision instead of YOLO tracking due to overwriting goalkeepers.
    """
    def __init__(self, model_path: str, classes: List[int], verbose: bool=True) -> None: 
        self.model = ultralytics.YOLO(model_path)
        self.classes = classes
        self.tracker = sv.ByteTrack()
        self.verbose = verbose

    def interpolate_ball_positions(self, ball_tracks: List[Dict]) -> List[Dict]:
        """
        If the ball is not detected in every frame, take the frames where it is detected and interpolate
        ball position in the frames between by drawing a line and simulate the position evenly along the line.
        """
        # tracker_id 1 for ball, {} if nothing found, at tracker_id 1, get bbox, [] if nothing found
        # {1: {"bbox": [....]}, ...
        ball_positions = [track.get(1, {}).get("bbox", []) for track in ball_tracks]
        df_ball_positions = pd.DataFrame(ball_positions, columns=["x1", "y1", "x2", "y2"])

        df_ball_positions = df_ball_positions.interpolate() # interpolate NaN values, estimation based on previous data
        df_ball_positions = df_ball_positions.bfill()       # --> edge case beginning: no previous data --> bfill (replace NaN with next following value)

        ball_positions = [{1: {"bbox": x}} for x in df_ball_positions.to_numpy().tolist()] # transform back

        return ball_positions

    def detect_frames(self, frames: List[np.ndarray], batch_size: int=20) -> List[ultralytics.engine.results.Results]:
        """
        List of frame predictions processed in batches to avoid memory issues.
        """
        batch_size=batch_size
        detections = []

        
        start_time = time.time()

        if self.verbose:
            logger.info(f"[Device: {get_device()}] Starting object detection at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        for i in range(0, len(frames), batch_size):
            frame_time = time.time()
            
            detections_batch = self.model.predict(source=frames[i:i+batch_size], conf=0.1, classes=self.classes, verbose=self.verbose, device=get_device())
            detections += detections_batch

            if self.verbose:
                logger.info(f"Processed frames {i} to {min(i+batch_size-1, len(frames))} in {time.time() - frame_time:.2f} seconds.")
        
        if self.verbose:
            logger.info(f"Detected objects in {len(frames)} frames in {time.time() - start_time:.2f} seconds.")
  
        return detections

    def get_object_tracks(self, frames: List[np.ndarray]) -> Dict[str, List[Dict]]:        
        detections = self.detect_frames(frames)

        # key: tracker_id, value: bbox, index: frame
        tracks = {
            "players": [],      # {tracker_id: {"bbox": [....]}, tracker_id: {"bbox": [....]}, tracker_id: {"bbox": [....]}, tracker_id: {"bbox": [....]}}  (same for referees and ball)
            "referees": [],     
            "ball": []
        }

        start_time = time.time()

        if self.verbose:
            logger.info(f"[Device: {get_device()}] Starting object tracking at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_switched = {v: k for k, v in cls_names.items()}       # swap keys and values, e.g. ball: 1 --> 1: ball for easier access

            # convert to supervision detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)      # xyxy bboxes
            
            # convert goalkeeper to player
            # goalkeepers might get predicted as players in some frames and that could cause tracking issues
            for index, class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == "goalkeeper":
                    detection_supervision.class_id[index] = cls_names_switched["player"]
            # before:
            # class_id=array([1, 2, 2, 2, 2, 3, 3]), tracker_id=None, data={'class_name': array(['goalkeeper', 'player', 'player', 'player', 'player', 'referee', 'referee'], dtype='<U7')}
            # after:          ^
            # class_id=array([2, 2, 2, 2, 2, 3, 3]), tracker_id=None, data={'class_name': array(['goalkeeper', 'player', 'player', 'player', 'player', 'referee', 'referee'], dtype='<U7')}
            
            # track objects
            detections_with_tracks = self.tracker.update_with_detections(detection_supervision)    # adds tracker object to detections, every object gets a unique tracker id
            # example:
            # class_id=array([2, 2, 2, 2, 2, 3, 3]), tracker_id=array([ 1,  2,  3,  4,  5,  6,  7]), data={'class_name': array(['player', 'player', 'player', 'player', 'player', 'referee', 'referee'], dtype='<U7')}

            tracks["players"].append({}) 
            tracks["referees"].append({})
            tracks["ball"].append({})
            
            for frame_detection in detections_with_tracks:
                # frame_detection: Detections (bboxes), mask, confidence, class_id, tracker_id, class_name
                bbox = frame_detection[0].tolist()
                class_id = frame_detection[3]
                tracker_id = frame_detection[4]

                # add object at class (players/referees/ball) at index (frame) with its unique tracker ID
                if class_id == cls_names_switched["player"]:
                    tracks["players"][frame_num][tracker_id] = {"bbox": bbox}

                if class_id == cls_names_switched["referee"]:
                    tracks["referees"][frame_num][tracker_id] = {"bbox": bbox}

            # no tracker for the ball as there is only one
            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                class_id = frame_detection[3]

                if class_id == cls_names_switched["ball"]:
                    tracks["ball"][frame_num][1] = {"bbox": bbox}   # ID 1 as there is only one ball

        if self.verbose:
            logger.info(f"Tracked objects in {len(frames)} frames in {time.time() - start_time:.2f} seconds.")

            separator = f"{'-'*10} [End of tracking] at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {'-'*10}"
            logger.info(separator)

        return tracks
    
    def add_position_to_tracks(self, tracks: Dict[str, List[Dict]]) -> None:
        for object, object_tracks in tracks.items():
            for frame_num, track_dict in enumerate(object_tracks):
                for tracker_id, track in track_dict.items():
                    bbox = track["bbox"]

                    if object == "ball":
                        position = get_center_of_bbox(bbox)
                    else:   # player
                        position = get_foot_position(bbox)

                    # add new key position
                    tracks[object][frame_num][tracker_id]["position"] = position

    def draw_annotations(self, frames: List[np.ndarray], tracks: Dict[str, List[Dict]], ball_possession: np.ndarray) -> List[np.ndarray]:   # TODO extra folder for custom drawings and then import?
        output_frames = []  # frames after changing the annotations
        
        for frame_num, frame in enumerate(frames):
            frame = frame.copy()       # don't change original

            player_dict = tracks["players"][frame_num]
            referee_dict = tracks["referees"][frame_num]
            ball_dict = tracks["ball"][frame_num]

            for tracker_id, player in player_dict.items():
                colour = player.get("team_colour", (255, 255, 255))         # get team colour if it exists, else white
                frame = ellipse(frame, player["bbox"], colour, tracker_id)

                if player.get("has_ball", False):
                    frame = triangle(frame, player["bbox"], (0, 0, 255))    # red triangle

            for _, referee in referee_dict.items():
                frame = ellipse(frame, referee["bbox"], (0, 255, 255))      # yellow ellipse

            for tracker_id, ball in ball_dict.items():
                frame = triangle(frame, ball["bbox"], (0, 255, 0))          # green triangle
                
            frame = ball_possession_box(frame_num, frame, ball_possession)

            output_frames.append(frame)

        return output_frames