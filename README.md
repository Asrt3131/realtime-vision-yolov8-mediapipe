# 👁️ Real-time Vision Suite — YOLOv8 + MediaPipe

<p align="center">
  <img src="screenshots/demo.png" alt="Real-time Vision Suite demo" width="900"/>
</p>

<p align="center">
  <b>Real-time object detection with YOLOv8 &nbsp;•&nbsp; Face landmark analysis with MediaPipe &nbsp;•&nbsp; Deployed on Streamlit</b>
</p>

<p align="center">
  <a href="https://realtime-vision-yolov8-mediapipe-emyqrahh3wfbgyvvycvcfq.streamlit.app/">
    <img src="https://static.streamlit.io/badges/streamlit_badge_black_white.svg" alt="Live Demo"/>
  </a>
  <a href="https://github.com/Asrt3131/realtime-vision-yolov8-mediapipe">
    <img src="https://img.shields.io/badge/GitHub-Source%20Code-black?style=flat-square&logo=github" alt="GitHub"/>
  </a>
  <img src="https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python" alt="Python"/>
  <img src="https://img.shields.io/badge/License-MIT-green?style=flat-square" alt="License"/>
</p>

---

## 🔗 Quick Links

| Resource | Link |
|----------|------|
| 🚀 **Live Demo (Try it now)** | **[https://realtime-vision-yolov8-mediapipe-emyqrahh3wfbgyvvycvcfq.streamlit.app/](https://realtime-vision-yolov8-mediapipe-emyqrahh3wfbgyvvycvcfq.streamlit.app/)** |
| 💻 **Source Code (GitHub)** | **[https://github.com/Asrt3131/realtime-vision-yolov8-mediapipe](https://github.com/Asrt3131/realtime-vision-yolov8-mediapipe)** |
| 📦 **Dataset / Database** | *(Coming soon — dataset link will be added here)* |

> 🔁 **Note:** Link to the dataset/database will be added soon. If you are the project owner, replace the placeholder in the **Dataset & Data Source** section below.

---

## ✨ Overview

**Real-time Vision Suite** is a dual-module computer vision application that combines two state-of-the-art models in a single Streamlit interface:

- **YOLOv8** (Ultralytics) — fast, accurate object detection across 80 COCO classes
- **MediaPipe Face Mesh** — 468 facial landmarks with head-pose and blink estimation

The app supports **image upload**, **video file processing**, and **live webcam capture**, with adjustable confidence/IoU thresholds, class filtering, and downloadable results (JSON + annotated image).

---

## 🎯 Features

### Module 1 — Object Detection (YOLOv8)
- ✅ 80 COCO classes with per-class color coding
- ✅ Adjustable **confidence** and **IoU (NMS)** thresholds
- ✅ **Class filter** — show only the objects you care about
- ✅ **Stats panel**: total objects, unique classes, top detected class
- ✅ **Export** as JSON (detections) and PNG (annotated image)
- ✅ Works on **images**, **videos** (frame sampling), and **webcam**

### Module 2 — Face Analysis (MediaPipe Face Mesh)
- ✅ **468 landmarks** rendered as a colored mesh
- ✅ **Head pose** estimation (yaw offset → "Forward" / "Turned")
- ✅ **Blink detection** using Eye Aspect Ratio (EAR)
- ✅ Multi-face support (up to 5 faces per frame)
- ✅ Per-face metrics in expandable cards
- ✅ JSON export

### General
- ⚡ Cached models (`@st.cache_resource`) for instant re-runs
- 🎨 Dark-themed UI with custom sidebar
- 📱 Responsive layout (wide mode)
- 🧩 Modular architecture — reusable `ObjectDetector` and `FaceAnalyzer` classes
- 🔌 Ready for **FastAPI** upgrade (see *Roadmap*)

---

## 🖼️ Screenshots

<p align="center">
  <img src="screenshots/object_detection.png" alt="Object detection" width="48%"/>
  <img src="screenshots/face_analysis.png" alt="Face analysis" width="48%"/>
</p>

> 📸 *Add your own screenshots to `screenshots/` — see the folder's README for instructions.*

---

## 🛠️ Tech Stack

| Layer | Technology | Version |
|-------|-----------|---------|
| Object Detection | [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) | 8.3.0 |
| Face Mesh | [MediaPipe](https://google.github.io/mediapipe/) | 0.10.14 |
| Image Processing | OpenCV (headless) | 4.10 |
| Frontend | Streamlit | 1.40 |
| Runtime | Python | 3.11 |

---

## 🚀 Run Locally

### Prerequisites
- Python 3.11 (⚠️ **3.12+ may break MediaPipe / Torch wheels — use 3.11**)
- pip
- (Optional) a GPU for faster inference
- On Linux: system package `libgl1` (required by OpenCV)

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/Asrt3131/realtime-vision-yolov8-mediapipe.git
cd realtime-vision-yolov8-mediapipe

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. (Linux only) install system libs required by OpenCV
sudo apt-get update && sudo apt-get install -y libgl1

# 4. Install dependencies
pip install -r requirements.txt

# 5. Run the app
streamlit run app.py
