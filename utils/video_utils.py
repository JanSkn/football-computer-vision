from typing import List, Union
import numpy as np
import cv2
import tempfile

def read_video(input: Union[str, bytes]) -> List[np.ndarray]:
    if isinstance(input, bytes):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as f:
            f.write(input)
            temp_filename = f.name
        cap = cv2.VideoCapture(temp_filename)
    elif isinstance(input, str):
        cap = cv2.VideoCapture(input)
    else:
        raise ValueError("Input data must be either bytes or a string file path.")
    
    frames = []
    while True:
        # ret: True/False if there is a next frame
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()

    return frames

def save_video(frames: List[np.ndarray], path: str, fps: int=24) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"avc1")    # codec for compressing the video
    out = cv2.VideoWriter(filename=path, fourcc=fourcc, fps=fps, frameSize=(frames[0].shape[1], frames[0].shape[0]))
    
    for frame in frames:
        out.write(frame)
    
    # close video file and release ressources
    out.release()   