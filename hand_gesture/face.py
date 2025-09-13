from typing import List, Tuple
import numpy as np
import cv2

try:
    import mediapipe as mp
except ImportError as exc:
    raise ImportError(
        "mediapipe is required. Install with `pip install mediapipe`."
    ) from exc


class FaceMeshDetector:
    """MediaPipe Face Mesh wrapper returning face landmarks as pixel coordinates."""

    def __init__(self,
                 max_num_faces: int = 1,
                 min_detection_confidence: float = 0.6,
                 min_tracking_confidence: float = 0.6):
        self._mp_face = mp.solutions.face_mesh
        self._mp_drawing = mp.solutions.drawing_utils
        self._mp_styles = mp.solutions.drawing_styles
        self.face_mesh = self._mp_face.FaceMesh(
            max_num_faces=max_num_faces,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def process(self, frame_bgr: np.ndarray) -> List[np.ndarray]:
        height, width = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)

        faces: List[np.ndarray] = []
        if results.multi_face_landmarks:
            for face_lms in results.multi_face_landmarks:
                coords = np.array(
                    [[lm.x * width, lm.y * height, lm.z] for lm in face_lms.landmark],
                    dtype=np.float32,
                )
                faces.append(coords)
        return faces


