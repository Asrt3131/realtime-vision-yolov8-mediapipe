import cv2
import mediapipe as mp
import numpy as np


class FaceAnalyzer:
    """
    MediaPipe Face Mesh wrapper با:
    - 468 landmark
    - تخمین yaw (چرخش افقی سر)
    - تشخیص پلک زدن (Eye Aspect Ratio)
    - رسم مش رنگی
    """

    # ایندکس نقاط کلیدی MediaPipe
    NOSE_TIP = 1
    CHIN = 152
    LEFT_EYE_L = 33
    LEFT_EYE_R = 133
    RIGHT_EYE_L = 362
    RIGHT_EYE_R = 263

    # EAR (Eye Aspect Ratio) — نقاط بالای پلک و پایین پلک
    LEFT_EYE_TOP = 159
    LEFT_EYE_BOTTOM = 145
    RIGHT_EYE_TOP = 386
    RIGHT_EYE_BOTTOM = 374

    EAR_THRESHOLD = 0.22  # زیر این مقدار = چشم بسته

    def __init__(self):
        self._fm = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=5,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._styles = mp.solutions.drawing_styles

    @staticmethod
    def _ear(top, bottom, left, right):
        """Eye Aspect Ratio ساده‌شده."""
        vert = np.linalg.norm(np.array([top.x - bottom.x, top.y - bottom.y]))
        horz = np.linalg.norm(np.array([left.x - right.x, left.y - right.y]))
        return float(vert / horz) if horz > 0 else 0.0

    def analyze(self, image_bgr):
        h, w = image_bgr.shape[:2]
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        results = self._fm.process(rgb)
        annotated = image_bgr.copy()
        faces_info = []

        if not results.multi_face_landmarks:
            return annotated, faces_info

        draw_utils = mp.solutions.drawing_utils
        for face_landmarks in results.multi_face_landmarks:
            # رسم مش
            draw_utils.draw_landmarks(
                annotated,
                face_landmarks,
                mp.solutions.face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=self._styles
                .get_default_face_mesh_contours_style(),
            )

            lm = face_landmarks.landmark

            # --- Head pose (yaw ساده) ---
            nose = lm[self.NOSE_TIP]
            left_eye = lm[self.LEFT_EYE_L]
            right_eye = lm[self.RIGHT_EYE_R]
            eye_center_x = (left_eye.x + right_eye.x) / 2
            yaw = (nose.x - eye_center_x) * 100  # درصد

            # --- Eye Aspect Ratio ---
            ear_left = self._ear(
                lm[self.LEFT_EYE_TOP], lm[self.LEFT_EYE_BOTTOM],
                lm[self.LEFT_EYE_L], lm[self.LEFT_EYE_R],
            )
            ear_right = self._ear(
                lm[self.RIGHT_EYE_TOP], lm[self.RIGHT_EYE_BOTTOM],
                lm[self.RIGHT_EYE_L], lm[self.RIGHT_EYE_R],
            )
            ear_avg = round((ear_left + ear_right) / 2, 3)
            blink = ear_avg < self.EAR_THRESHOLD

            # --- Bounding box تقریبی از نقاط صورت ---
            xs = [p.x for p in lm]
            ys = [p.y for p in lm]
            x1, x2 = int(min(xs) * w), int(max(xs) * w)
            y1, y2 = int(min(ys) * h), int(max(ys) * h)
            color = (0, 200, 255) if blink else (0, 255, 0)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 1)

            faces_info.append({
                "yaw_offset_percent": round(yaw, 2),
                "head_turned": abs(yaw) > 15,
                "ear": ear_avg,
                "blink": blink,
                "bbox_xyxy": [x1, y1, x2, y2],
            })

        return annotated, faces_info
