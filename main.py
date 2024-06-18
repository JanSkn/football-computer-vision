from typing import Union, List
from utils import read_video, save_video
from trackers import Tracker
from team_assignment import TeamAssigner

def process_video(data: Union[str, bytes], classes: List[int]) -> None:
    frames = read_video(data)

    tracker = Tracker("models/best.pt", classes=classes)
    tracks = tracker.get_object_tracks(frames)

    team_assigner = TeamAssigner()
    team_assigner.get_teams(frames, tracks)

    output = tracker.annotations(frames, tracks)
    save_video(output, "output/output.mp4")

if __name__ == "__main__":
    process_video("demos/demo1.mp4", [0, 1, 2, 3])