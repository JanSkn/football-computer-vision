from typing import Union, List
from utils import read_video, save_video
from trackers import Tracker
from team_assignment import TeamAssigner
from player_ball_assignment import PlayerBallAssigner

def process_video(data: Union[str, bytes], classes: List[int]) -> None:
    frames, fps, _, _ = read_video(data)

    tracker = Tracker("models/best.pt", classes)
    tracks = tracker.get_object_tracks(frames)

    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    team_assigner = TeamAssigner()
    team_assigner.get_teams(frames, tracks)

    player_assigner = PlayerBallAssigner()
    player_assigner.get_player_and_possession(tracks)

    output = tracker.annotations(frames, tracks, player_assigner.ball_possession)
    save_video(output, "output/output.mp4", fps)

if __name__ == "__main__":
    process_video("demos/demo1.mp4", [0, 1, 2, 3])