from typing import Union
from utils import read_video, save_video
from trackers import Tracker
from team_assignment import TeamAssigner

def main():
    frames = read_video("demos/short.mp4")

    tracker = Tracker("models/best.pt")
    tracks = tracker.get_object_tracks(frames)

    team_assigner = TeamAssigner()
    team_assigner.get_teams(frames, tracks)

    output = tracker.annotations(frames, tracks)
    save_video(output, "output/output.mp4")

def process_video(data: Union[str, bytes]) -> None:
    frames = read_video(data)

    tracker = Tracker("models/best.pt")
    tracks = tracker.get_object_tracks(frames)

    team_assigner = TeamAssigner()
    team_assigner.get_teams(frames, tracks)

    output = tracker.annotations(frames, tracks)
    save_video(output, "output/output.mp4")

if __name__ == "__main__":
    main()