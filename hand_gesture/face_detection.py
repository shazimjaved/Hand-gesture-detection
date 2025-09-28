"""
Face detection module using MediaPipe FaceMesh for facial landmark detection.
"""

from typing import List, Optional, Tuple
import numpy as np
import cv2

try:
    import mediapipe as mp
except ImportError as exc:
    raise ImportError(
        "mediapipe is required. Install with `pip install mediapipe`."
    ) from exc


class FaceDetector:
    """Wrapper around MediaPipe FaceMesh for facial landmark detection."""

    def __init__(
        self,
        static_image_mode: bool = False,
        max_num_faces: int = 1,
        refine_landmarks: bool = True,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        self._mp_face_mesh = mp.solutions.face_mesh
        self._mp_drawing = mp.solutions.drawing_utils
        self._mp_styles = mp.solutions.drawing_styles
        
        self.face_mesh = self._mp_face_mesh.FaceMesh(
            static_image_mode=static_image_mode,
            max_num_faces=max_num_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def process(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Process a BGR image and return:
        - RGB image used for processing
        - List of landmark arrays (468x3) in pixel coordinates for each face
        """
        height, width = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        landmark_list: List[np.ndarray] = []

        results = self.face_mesh.process(frame_rgb)
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                coords = np.array(
                    [[lm.x * width, lm.y * height, lm.z] for lm in face_landmarks.landmark],
                    dtype=np.float32,
                )
                landmark_list.append(coords)

        return frame_rgb, landmark_list
