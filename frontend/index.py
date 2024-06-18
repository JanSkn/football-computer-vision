import streamlit as st
import time
import os
import sys
sys.path.append(os.path.abspath("."))
from main import process_video

st.set_page_config(page_title="Football Analysis")

# menu
st.sidebar.title("Settings")

st.sidebar.header("Options")
players = st.sidebar.toggle("Track Players", value=True)
goalkeepers = st.sidebar.toggle("Track Goalkeepers", value=True)
referees = st.sidebar.toggle("Track Referees", value=True)
ball = st.sidebar.toggle("Track Ball", value=True)
keypoints = st.sidebar.toggle("Show Keypoints", value=True)
speed = st.sidebar.toggle("Show Players' Speed", value=True)

# select classes to track

# data.yaml class IDs
# ball: 0, goalkeeper: 1, player: 2, referee: 3
options = {0: ball, 1: goalkeepers, 2: players, 3: referees}
classes = [key for key, value in options.items() if value is True]

st.sidebar.markdown("***")

st.sidebar.header("Video Source")

st.sidebar.subheader("Demo")

st.sidebar.write("Choose from 3 demo videos.")

uploaded_video = None
demo_video = None
start_analysis = None
processed = False

demo = st.sidebar.toggle("Demo", value=False)

if demo:
    videos = [
    "demos/demo1.mp4",
    "demos/demo2.mp4",
    "demos/demo3.mp4"
    ]

    demo_video = st.sidebar.radio("Select Video", videos)

    #preview demo video
    st.sidebar.video(demo_video)
    
    if demo_video:
        with open(demo_video, "rb") as f:
            demo_video_bytes = f.read()

    start_analysis = st.sidebar.button("Start Analysis", key="demo")

    if start_analysis:
        with st.spinner("Processing ..."):
            process_video(demo_video_bytes, classes)
            processed = True
        placeholder = st.empty()
        with placeholder.container():
            st.success("Video processing complete.")
            time.sleep(3)
        placeholder.empty()

st.sidebar.write("\n")

st.sidebar.subheader("Video Upload")

uploaded_video = st.sidebar.file_uploader("Select a video file.", type=["mp4"])

if uploaded_video:
    st.sidebar.video(uploaded_video)
    st.sidebar.write("Uploaded video:", uploaded_video.name)
    start_analysis = st.sidebar.button("Start Analysis", key="upload")

    if start_analysis:
        with st.spinner("Processing ..."):
            process_video(uploaded_video.read(), classes) 
            processed = True
        placeholder = st.empty()
        with placeholder.container():
            st.success("Video processing complete.")
            time.sleep(3)
        placeholder.empty()

# main page
st.title("MatchVision - automated analysis")
st.subheader("Computer Vision & Deep Learning")

tab1, tab2, tab3 = st.tabs(["Usage", "Results", "Logs"])

with tab1:
    st.write("To use the automated analysis, follow these steps:")
    st.markdown("""
    1. Select the desired output options.
    2. Upload a video or select a demo video. 
    3. Click on **Start Analysis**.
    4. Go to the tab **Results** to see the output video.
                
    For best results, the video should not contain multiple camera perspectives.
    """)

with tab2:
    if processed:
        st.video("output/output.mp4")

with tab3:
    log_files = "model_logs/tracking.log"

    try:
        with open(log_files, "r") as log_file:
            log_contents = log_file.read()
        st.text_area("Tracking logs", log_contents, height=450)
    except FileNotFoundError:
        st.error(f"Log file '{log_files}' not found.")