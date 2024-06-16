from typing import List, Dict, Tuple
import numpy as np
import ultralytics
import supervision as sv
import cv2
# import sys
# sys.path.append("../")
from utils import get_center_of_bbox, get_bbox_dimensions

class Tracker:
    """
    Byte tracker. 
    Tracking persons by close bounding box in next frame combined with movement and visual features like shirt colour.
    Assigning bounding boxes unique IDs.
    Predicting and then tracking with supervision instead of YOLO tracking due to overwriting goalkeepers.
    """
    def __init__(self, model_path: str) -> None: # TODO , classes: List[int] for selection in frontend, add classes=self.classes in predict
        self.model = ultralytics.YOLO(model_path)
        #self.classes = classes
        self.tracker = sv.ByteTrack()

    def detect_frames(self, frames: List[np.ndarray], batch_size: int=20) -> List[ultralytics.engine.results.Results]:
        """
        List of frame predictions processed in batches to avoid memory issues.
        """
        batch_size=batch_size
        detections = []

        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(source=frames[i:i+batch_size], conf=0.1)
            detections += detections_batch
            
        return detections

    def get_object_tracks(self, frames: List[np.ndarray]) -> Dict[str, List[Dict]]:
        detections = self.detect_frames(frames)

        # key: tracker_id, value: bbox, index: frame
        tracks = {
            "players": [],      # {tracker_id: {"bbox": [....]}, tracker_id: {"bbox": [....]}, tracker_id: {"bbox": [....]}, tracker_id: {"bbox": [....]}}  (same for referees and ball)
            "referees": [],     
            "ball": []
        }

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

        return tracks
    
    def ellipse(self, frame: np.ndarray, bbox: List[float], colour: Tuple[int, int, int], tracker_id: int=None):
        # xyxy bboxes --> y2 at last index
        y2 = int(bbox[3])   # ellipse should be below the player

        x_center, _ = get_center_of_bbox(bbox)
        width, _ = get_bbox_dimensions(bbox)

        cv2.ellipse(frame, 
                    center=(x_center, y2), 
                    axes=(width,           # length of major axis
                          0.35*width),     # length of minor axis
                    angle=0,                    # rotation of the ellipse
                    startAngle=-45,             # start and end: upper part of ellipse not drawn
                    endAngle=235,    
                    color=colour, 
                    thickness=2)
        
        if tracker_id:
            rect_width = 40
            rect_height = 20
            rect_x1 = x_center - rect_width // 2
            rect_x2 = x_center + rect_width // 2
            rect_y1 = y2 - rect_height // 2 + 15       # padding
            rect_y2 = y2 + rect_height // 2 + 15       # padding

            cv2.rectangle(frame, 
                        pt1=(rect_x1, rect_y1),   # top left corner
                        pt2=(rect_x2, rect_y2),   # bottom right corner
                        color=colour,
                        thickness=cv2.FILLED)

            # size of the text for centering
            text = str(tracker_id)
            font_face = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2

            (text_width, text_height), _ = cv2.getTextSize(text, font_face, font_scale, thickness)
            text_x = rect_x1 + (rect_width - text_width) // 2
            text_y = rect_y1 + (rect_height + text_height) // 2

            cv2.putText(frame, 
                        text=text, 
                        org=(text_x, text_y),  # bottom left corner of the text
                        fontFace=font_face, 
                        fontScale=font_scale, 
                        color=(0, 0, 0), 
                        thickness=thickness)

        return frame
    
    def triangle(self, frame: np.ndarray, bbox: List[float], colour: Tuple[int, int, int]):
        y = int(bbox[1])    # y1 --> top of the ball
        x, _ = get_center_of_bbox(bbox)

        triangle_points = np.array([
            [x, y],                 # bottom corner
            [x - 8, y - 15],       # top left corner
            [x + 8, y - 15],       # top right corner
        ])   

        cv2.drawContours(frame, contours=[triangle_points], contourIdx=0, color=colour, thickness=cv2.FILLED)   
        cv2.drawContours(frame, contours=[triangle_points], contourIdx=0, color=(0, 0, 0), thickness=2)   # border  

        return frame        

    def annotations(self, frames: List[np.ndarray], tracks: Dict[str, List[Dict]]) -> List[np.ndarray]:   # TODO extra folder for custom drawings and then import?
        output_frames = []  # frames after changing the annotations
        
        for frame_num, frame in enumerate(frames):
            frame = frame.copy()       # don't change original

            player_dict = tracks["players"][frame_num]
            referee_dict = tracks["referees"][frame_num]
            ball_dict = tracks["ball"][frame_num]

            for tracker_id, player in player_dict.items():
                frame = self.ellipse(frame, player["bbox"], (255, 255, 255), tracker_id)

            for _, referee in referee_dict.items():
                frame = self.ellipse(frame, referee["bbox"], (0, 255, 255))

            for tracker_id, ball in ball_dict.items():
                frame = self.triangle(frame, ball["bbox"], (0, 255, 0))
                
            output_frames.append(frame)

        return output_frames