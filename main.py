from typing import Union, List
from utils import read_video, save_video
from trackers import Tracker
from team_assignment import TeamAssigner
from player_ball_assignment import PlayerBallAssigner
from camera_movement import CameraMovementEstimator

def process_video(data: Union[str, bytes], classes: List[int]) -> None:
    frames, fps, _, _ = read_video(data)

    tracker = Tracker("models/best.pt", classes)
    tracks = tracker.get_object_tracks(frames)

    camera_movement_estimator = CameraMovementEstimator(frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(frames)

    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    team_assigner = TeamAssigner()
    team_assigner.get_teams(frames, tracks)

    player_assigner = PlayerBallAssigner()
    player_assigner.get_player_and_possession(tracks)

    output = tracker.draw_annotations(frames, tracks, player_assigner.ball_possession)
    output = camera_movement_estimator.draw_camera_movement(output, camera_movement_per_frame)
    
    save_video(output, "output/output.mp4", fps)

if __name__ == "__main__":
    process_video("demos/demo1.mp4", [0, 1, 2, 3])