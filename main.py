from utils import read_video, save_video
from trackers import Tracker

def main():
    frames = read_video("videos/short.mp4")

    tracker = Tracker("models/best.pt")
    tracks = tracker.get_object_tracks(frames)

    output = tracker.annotations(frames, tracks)

    save_video(output, "output/output.mp4")

if __name__ == "__main__":
    main()