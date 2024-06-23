from typing import List, Union, Tuple
import numpy as np
import cv2
import time
from datetime import datetime
import logging
import tempfile

file_handler = logging.FileHandler("logs/memory_access.log")
file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

logger = logging.getLogger("memory_access")
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.addHandler(stream_handler)

def read_video(input: Union[str, bytes], verbose: bool=True) -> Tuple[List[np.ndarray], int, int, str]:
    start_time = time.time()

    if isinstance(input, bytes):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as f:
            f.write(input)
            temp_filename = f.name
        cap = cv2.VideoCapture(temp_filename)
    elif isinstance(input, str):
        cap = cv2.VideoCapture(input)
    else:
        raise ValueError("Input data must be either bytes or a string file path.")
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    frames = []
    while True:
        # ret: True/False if there is a next frame
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])

    cap.release()

    if verbose:
            logger.info(f"Reading input video from memory in {time.time() - start_time:.2f} seconds.")

    return frames, fps, fourcc, codec

def save_video(frames: List[np.ndarray], path: str, fps: int=24, verbose: bool=True) -> None:
    start_time = time.time()

    fourcc = cv2.VideoWriter_fourcc(*"avc1")    # codec for compressing the video
    out = cv2.VideoWriter(filename=path, fourcc=fourcc, fps=fps, frameSize=(frames[0].shape[1], frames[0].shape[0]))
    
    for frame in frames:
        out.write(frame)
    
    # close video file and release ressources
    out.release()   

    if verbose:
            logger.info(f"Saving video in {time.time() - start_time:.2f} seconds.")