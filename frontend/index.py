import streamlit as st
import os

st.set_page_config(page_title="Football Analysis")

# menu
st.sidebar.title("Settings")

st.sidebar.subheader("Options")
players = st.sidebar.toggle("Track Players", value=True)
goalkeepers = st.sidebar.toggle("Track Goalkeepers", value=True)
ball = st.sidebar.toggle("Track Ball", value=True)
keypoints = st.sidebar.toggle("Show Keypoints", value=True)
speed = st.sidebar.toggle("Show Players' Speed", value=True)

st.sidebar.markdown("***")

st.sidebar.subheader("Video Upload")
video_file = st.sidebar.file_uploader("Select a video file.", type=["mp4", "mov", "avi", "mkv"])

if video_file is not None:
    st.sidebar.video(video_file)
    st.sidebar.write("Uploaded video:", video_file.name)

# main page
st.title("MatchVision - automated analysis")
st.subheader("Computer Vision & Deep Learning")

tab1, tab2 = st.tabs(["Usage", "Results"])

with tab1:
    st.write("To use the automated analysis, follow these steps:")
    st.markdown("""
    1. Select the desired output options
    2. Upload a video. Maximum video size is 200MB. 
    3. Go to the tab **Results** to see the output video.
    4. Download your final video.
                
    For best results, the video should not contain multiple camera perspectives.
    """)

with tab2:
    video_file = open("videos/019d5b34_1.mp4", "rb")
    video_bytes = video_file.read()
    st.video(video_bytes)