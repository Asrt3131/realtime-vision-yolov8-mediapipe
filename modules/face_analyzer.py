import cv2
import mediapipe as mp
import numpy as np

class FaceAnalyzer:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=5,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
        # نقاط کلیدی برای تخمین حالت سر (بینی، چانه، گوشه چشم‌ها)
        self.NOSE_TIP = 1
        self.CHIN = 152
        self.LEFT_EYE = 33
        self.RIGHT_EYE = 263

    def analyze(self, image_bgr):
        h, w = image_bgr.shape[:2]
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        annotated = image_bgr.copy()
        faces_info = []

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # رسم نقاط کلیدی
                mp.solutions.drawing_utils.draw_landmarks(
                    annotated, face_landmarks,
                    self.mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp.solutions.drawing_styles
                    .get_default_face_mesh_contours_style()
                )
                # تخمین ساده حالت سر (yaw تقریبی)
                nose = face_landmarks.landmark[self.NOSE_TIP]
                left_eye = face_landmarks.landmark[self.LEFT_EYE]
                right_eye = face_landmarks.landmark[self.RIGHT_EYE]

                # نسبت فاصله بینی تا چشم‌ها (نشان‌دهنده چرخش افقی سر)
                eye_center_x = (left_eye.x + right_eye.x) / 2
                yaw_offset = (nose.x - eye_center_x) * 100  # درصد

                faces_info.append({
                    "yaw_offset_percent": round(yaw_offset, 2),
                    "head_turned": abs(yaw_offset) > 15  # آستانه 15٪
                })
        return annotated, faces_info