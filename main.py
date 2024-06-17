from utils import read_video, save_video
from trackers import Tracker
from team_assignment import TeamAssigner

def main():
    frames = read_video("videos/short.mp4")

    tracker = Tracker("models/best.pt", verbose=False)
    tracks = tracker.get_object_tracks(frames)

    team_assigner = TeamAssigner()
    team_assigner.get_teams(frames, tracks)

    output = tracker.annotations(frames, tracks)
    save_video(output, "output/output.mp4")

if __name__ == "__main__":
    main()