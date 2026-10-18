import io
import json
import time
from datetime import datetime

import cv2
import numpy as np
import streamlit as st

from modules.object_detector import ObjectDetector, COCO_CLASSES
from modules.face_analyzer import FaceAnalyzer

# ---------------- Page config ----------------
st.set_page_config(
    page_title="Real-time Vision Suite",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("👁️ Real-time Vision Suite")
st.caption(
    "YOLOv8 Object Detection + MediaPipe Face Analysis | "
    "Built with ❤️ by A_srt343"
)

# ---------------- Cached models ----------------
@st.cache_resource(show_spinner="Loading YOLOv8 + MediaPipe...")
def load_models():
    return ObjectDetector("yolov8n.pt"), FaceAnalyzer()

obj_detector, face_analyzer = load_models()

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("⚙️ Settings")
    input_mode = st.radio(
        "Input source",
        ["📷 Image upload", "🎥 Video file", "🖥️ Webcam (camera_input)"],
        index=0,
    )
    st.divider()
    st.subheader("Detection parameters")
    conf = st.slider("Confidence", 0.05, 0.95, 0.40, 0.05)
    iou = st.slider("IoU (NMS)", 0.05, 0.95, 0.50, 0.05)
    st.divider()
    st.subheader("Filter classes (optional)")
    selected_classes = st.multiselect(
        "Show only:",
        options=COCO_CLASSES,
        default=[],
        help="Empty = show all 80 COCO classes",
    )
    st.divider()
    st.caption("Made with Streamlit, YOLOv8 & MediaPipe")

classes_filter = (
    [obj_detector.names_inv[c] for c in selected_classes if c in obj_detector.names_inv]
    if selected_classes else None
)

# ---------------- Tabs ----------------
tab1, tab2 = st.tabs(["🎯 Object Detection", "😊 Face Analysis"])

# ============================================================
# TAB 1 — Object Detection
# ============================================================
with tab1:
    st.subheader("Object Detection (YOLOv8)")

    def run_object_on_frame(frame_bgr):
        annotated, dets, stats = obj_detector.detect(
            frame_bgr, conf=conf, iou=iou, classes_filter=classes_filter
        )
        return annotated, dets, stats

    # ---------- Image mode ----------
    if input_mode == "📷 Image upload":
        uploaded = st.file_uploader(
            "Upload an image", type=["jpg", "jpeg", "png"], key="obj_img"
        )
        if uploaded:
            buf = np.frombuffer(uploaded.read(), np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            annotated, dets, stats = run_object_on_frame(img)

            c1, c2 = st.columns(2)
            c1.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Original")
            c2.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                     caption=f"Detected — {stats['total']} object(s)")

            m1, m2, m3 = st.columns(3)
            m1.metric("Total objects", stats["total"])
            m2.metric("Unique classes", len(stats["by_class"]))
            m3.metric("Top class",
                      max(stats["by_class"], key=stats["by_class"].get)
                      if stats["by_class"] else "—")

            if dets:
                st.dataframe(dets, use_container_width=True)
                st.download_button(
                    "⬇️ Download JSON",
                    data=json.dumps(dets, indent=2),
                    file_name=f"detections_{datetime.now():%Y%m%d_%H%M%S}.json",
                    mime="application/json",
                )
                ok, buf_png = cv2.imencode(".png", annotated)
                if ok:
                    st.download_button(
                        "⬇️ Download annotated image",
                        data=buf_png.tobytes(),
                        file_name="annotated.png",
                        mime="image/png",
                    )

    # ---------- Video mode ----------
    elif input_mode == "🎥 Video file":
        uploaded = st.file_uploader(
            "Upload a short video", type=["mp4", "mov", "avi"], key="obj_vid"
        )
        if uploaded:
            tmp_path = f"/tmp/_upload_{int(time.time())}.mp4"
            with open(tmp_path, "wb") as f:
                f.write(uploaded.read())
            cap = cv2.VideoCapture(tmp_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
            stride = max(1, total_frames // 30)  # ~30 frames preview
            frame_placeholder = st.empty()
            stats_placeholder = st.empty()
            idx = 0
            shown = 0
            t0 = time.time()
            while cap.isOpened() and shown < 30:
                ret, frame = cap.read()
                if not ret:
                    break
                if idx % stride == 0:
                    annotated, dets, stats = run_object_on_frame(frame)
                    frame_placeholder.image(
                        cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                        caption=f"Frame {idx} — {stats['total']} objects",
                    )
                    stats_placeholder.json(stats)
                    shown += 1
                idx += 1
            cap.release()
            elapsed = time.time() - t0
            st.success(f"Processed {shown} preview frames in {elapsed:.2f}s "
                       f"({shown / max(elapsed, 1e-6):.1f} fps preview)")

    # ---------- Webcam ----------
    else:
        st.info("📷 Take a photo with your camera. On Streamlit Cloud this works if the browser allows camera access.")
        cam = st.camera_input("Capture")
        if cam:
            buf = np.frombuffer(cam.read(), np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            annotated, dets, stats = run_object_on_frame(img)
            c1, c2 = st.columns(2)
            c1.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Captured")
            c2.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                     caption=f"{stats['total']} object(s)")
            if dets:
                st.dataframe(dets, use_container_width=True)

# ============================================================
# TAB 2 — Face Analysis
# ============================================================
with tab2:
    st.subheader("Face Analysis (MediaPipe Face Mesh)")

    face_mode = st.radio(
        "Face input", ["Image", "Camera"], horizontal=True, key="face_mode"
    )

    def run_face_on_frame(frame_bgr):
        return face_analyzer.analyze(frame_bgr)

    if face_mode == "Image":
        up2 = st.file_uploader(
            "Upload a face image", type=["jpg", "jpeg", "png"], key="face_img"
        )
        if up2:
            buf = np.frombuffer(up2.read(), np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            annotated, faces = run_face_on_frame(img)

            c1, c2 = st.columns(2)
            c1.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Original")
            c2.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                     caption=f"{len(faces)} face(s) — {'blinking' if any(f['blink'] for f in faces) else 'normal'}")

            if not faces:
                st.warning("No face detected. Try a clearer, frontal photo.")
            else:
                for i, f in enumerate(faces, 1):
                    with st.expander(f"🧑 Face {i}", expanded=(i == 1)):
                        a, b, c = st.columns(3)
                        a.metric("Yaw offset", f"{f['yaw_offset_percent']}%",
                                 delta="Turned" if f["head_turned"] else "Forward",
                                 delta_color="inverse" if f["head_turned"] else "normal")
                        b.metric("EAR", f"{f['ear']}",
                                 delta="Blink" if f["blink"] else "Open",
                                 delta_color="inverse" if f["blink"] else "normal")
                        c.metric("BBox (px)", f"{f['bbox_xyxy'][2]-f['bbox_xyxy'][0]}×"
                                              f"{f['bbox_xyxy'][3]-f['bbox_xyxy'][1]}")
                st.download_button(
                    "⬇️ Download face JSON",
                    data=json.dumps(faces, indent=2),
                    file_name=f"faces_{datetime.now():%Y%m%d_%H%M%S}.json",
                    mime="application/json",
                )

    else:
        cam2 = st.camera_input("Capture face", key="face_cam")
        if cam2:
            buf = np.frombuffer(cam2.read(), np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            annotated, faces = run_face_on_frame(img)
            c1, c2 = st.columns(2)
            c1.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Captured")
            c2.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                     caption=f"{len(faces)} face(s)")
            for i, f in enumerate(faces, 1):
                st.write(f"**Face {i}** → yaw={f['yaw_offset_percent']}%, "
                         f"EAR={f['ear']}, blink={f['blink']}")

# ---------------- Footer with credits ----------------
st.divider()
st.markdown(
    """
    <div style="text-align:center; color:#888; font-size:0.9em;">
        Built with ❤️ by <b>A_srt343</b><br>
        Instagram: <a href="https://instagram.com/project_srt343" target="_blank">
        @project_srt343</a>
    </div>
    """,
    unsafe_allow_html=True,
)
