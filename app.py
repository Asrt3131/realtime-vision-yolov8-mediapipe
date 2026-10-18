import streamlit as st
import cv2
import numpy as np
from modules.object_detector import ObjectDetector
from modules.face_analyzer import FaceAnalyzer

st.set_page_config(
    page_title="Real-time Vision Suite",
    page_icon="👁️",
    layout="wide"
)

st.title("👁️ Real-time Vision Suite")
st.caption("YOLOv8 Object Detection + MediaPipe Face Analysis | Built with ❤️ by A_srt343")

@st.cache_resource
def load_models():
    return ObjectDetector("yolov8n.pt"), FaceAnalyzer()

obj_detector, face_analyzer = load_models()

tab1, tab2 = st.tabs(["🎯 Object Detection (YOLOv8)", "😊 Face Analysis (MediaPipe)"])

with tab1:
    st.header("Object Detection")
    col1, col2 = st.columns([2, 1])
    with col2:
        conf = st.slider("Confidence", 0.1, 0.9, 0.4, 0.05)
        iou = st.slider("IoU (NMS)", 0.1, 0.9, 0.5, 0.05)
    with col1:
        uploaded = st.file_uploader("Upload image", type=["jpg", "png", "jpeg"], key="obj")
        if uploaded:
            img = cv2.imdecode(np.frombuffer(uploaded.read(), np.uint8), cv2.IMREAD_COLOR)
            annotated, dets = obj_detector.detect(img, conf, iou)
            c1, c2 = st.columns(2)
            c1.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Original")
            c2.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption=f"Detected ({len(dets)} objects)")
            if dets:
                st.dataframe(dets, use_container_width=True)

with tab2:
    st.header("Face Analysis (Head Pose & Landmarks)")
    uploaded2 = st.file_uploader("Upload face image", type=["jpg", "png", "jpeg"], key="face")
    if uploaded2:
        img = cv2.imdecode(np.frombuffer(uploaded2.read(), np.uint8), cv2.IMREAD_COLOR)
        annotated, faces = face_analyzer.analyze(img)
        c1, c2 = st.columns(2)
        c1.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Original")
        c2.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption=f"Detected {len(faces)} face(s)")
        for i, f in enumerate(faces):
            st.metric(f"Face {i+1} - Yaw Offset", f"{f['yaw_offset_percent']}%",
                      delta="Turned" if f['head_turned'] else "Forward",
                      delta_color="inverse" if f['head_turned'] else "normal")