from typing import List, Union, Tuple
import numpy as np
import cv2
import tempfile
import os
import platform
import sys
sys.path.append(os.path.abspath(".."))

def set_opencv_videoio_dll_path():
    system = platform.system()
    if system == "Windows":
        dll_path = "dll_files/openh264-1.8.0-win64.dll"
    elif system == "Darwin":  # macOS
        dll_path = "dll_files/openh264.dylib"
    elif system == "Linux":
        dll_path = "dll_files/libopenh264.so.1.8.0"
    else:
        raise OSError("Unsupported operating system")

    os.environ["OPENCV_VIDEOIO_DLL_PATH"] = dll_path

#set_opencv_videoio_dll_path()

def read_video(input: Union[str, bytes]) -> Tuple[List[np.ndarray], int, int, str]:
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

    return frames, fps, fourcc, codec

def save_video(frames: List[np.ndarray], path: str, fps: int=24) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"avc1")    # codec for compressing the video
    out = cv2.VideoWriter(filename=path, fourcc=fourcc, fps=fps, frameSize=(frames[0].shape[1], frames[0].shape[0]))
    
    for frame in frames:
        out.write(frame)
    
    # close video file and release ressources
    out.release()   