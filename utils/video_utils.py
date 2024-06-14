from typing import List
import numpy as np
import cv2

def read_video(path: str) -> List[np.ndarray]:
    cap = cv2.VideoCapture(filename=path)    # read video file
    frames = []                     # store all frames
    
    while True:
        # ret: True/False if there is a next frame
        ret, frame = cap.read()

        if not ret:
            break
        frames.append(frame)

    return frames

def save_video(frames: List[np.ndarray], path: str, fps: int=24) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")    # codec for compressing the video
    out = cv2.VideoWriter(filename=path, fourcc=fourcc, fps=fps, frameSize=(frames[0].shape[1], frames[0].shape[0]))
    
    for frame in frames:
        out.write(frame)
    
    # close video file and release ressources
    out.release()   