from typing import List, Optional, Tuple
import numpy as np
import cv2

try:
    import mediapipe as mp
except ImportError as exc:
    raise ImportError(
        "mediapipe is required. Install with `pip install mediapipe`."
    ) from exc

try:
    from ultralytics import YOLO  # type: ignore
    _ULTRALYTICS_AVAILABLE = True
except Exception:
    _ULTRALYTICS_AVAILABLE = False


class HandDetector:
    """Wrapper around MediaPipe Hands returning landmarks in pixel coordinates."""

    def __init__(
        self,
        static_image_mode: bool = False,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.6,
        min_tracking_confidence: float = 0.6,
        use_yolo: bool = False,
        yolo_model_path: Optional[str] = None,
    ):
        self._mp_hands = mp.solutions.hands
        self._mp_drawing = mp.solutions.drawing_utils
        self._mp_styles = mp.solutions.drawing_styles
        self.hands = self._mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            model_complexity=1,
        )
        # YOLO optional hand detector (expects a hand-detection model)
        self._use_yolo = bool(use_yolo and _ULTRALYTICS_AVAILABLE and yolo_model_path)
        self._yolo_model = None
        if self._use_yolo:
            try:
                self._yolo_model = YOLO(yolo_model_path)  # custom hand model path
            except Exception:
                # Fallback: disable YOLO if model fails to load
                self._use_yolo = False

    def process(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray], List[str]]:
        """
        Process a BGR image and return:
        - RGB image used for processing
        - List of landmarks arrays (21x3) in pixel coordinates
        - List of handedness labels ('Left' | 'Right')
        """
        height, width = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        landmark_list: List[np.ndarray] = []
        handedness_list: List[str] = []

        if self._use_yolo and self._yolo_model is not None:
            # Run YOLO to get hand ROIs, then run MediaPipe on each crop
            try:
                yolo_results = self._yolo_model.predict(source=frame_bgr, verbose=False, imgsz=640, conf=0.35)
                boxes = []
                if yolo_results and len(yolo_results) > 0:
                    for b in yolo_results[0].boxes:  # type: ignore[attr-defined]
                        xyxy = b.xyxy[0].tolist()  # [x1, y1, x2, y2]
                        x1, y1, x2, y2 = map(int, xyxy)
                        # Clamp to image bounds
                        x1 = max(0, min(x1, width - 1))
                        y1 = max(0, min(y1, height - 1))
                        x2 = max(0, min(x2, width))
                        y2 = max(0, min(y2, height))
                        if x2 > x1 and y2 > y1:
                            boxes.append((x1, y1, x2, y2))

                # If YOLO found boxes, run MediaPipe per ROI
                for (x1, y1, x2, y2) in boxes:
                    crop = frame_bgr[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue
                    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    crop_results = self.hands.process(crop_rgb)
                    if crop_results.multi_hand_landmarks:
                        for hand_landmarks in crop_results.multi_hand_landmarks:
                            cw, ch = (x2 - x1), (y2 - y1)
                            coords = np.array(
                                [
                                    [lm.x * cw + x1, lm.y * ch + y1, lm.z]
                                    for lm in hand_landmarks.landmark
                                ],
                                dtype=np.float32,
                            )
                            landmark_list.append(coords)
                    if crop_results.multi_handedness:
                        for hand_label in crop_results.multi_handedness:
                            handedness_list.append(hand_label.classification[0].label)

                # If YOLO produced nothing, fall back to full-frame MediaPipe
                if not landmark_list:
                    results = self.hands.process(frame_rgb)
                    self._collect_fullframe_results(results, width, height, landmark_list, handedness_list)

            except Exception:
                # On any YOLO error, gracefully fall back to full-frame MediaPipe
                results = self.hands.process(frame_rgb)
                self._collect_fullframe_results(results, width, height, landmark_list, handedness_list)
        else:
            # Standard full-frame MediaPipe
            results = self.hands.process(frame_rgb)
            self._collect_fullframe_results(results, width, height, landmark_list, handedness_list)

        return frame_rgb, landmark_list, handedness_list

    def _collect_fullframe_results(self, results, width: int, height: int,
                                   landmark_list: List[np.ndarray], handedness_list: List[str]) -> None:
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                coords = np.array(
                    [[lm.x * width, lm.y * height, lm.z] for lm in hand_landmarks.landmark],
                    dtype=np.float32,
                )
                landmark_list.append(coords)
        if getattr(results, 'multi_handedness', None):
            for hand_label in results.multi_handedness:
                handedness_list.append(hand_label.classification[0].label)

    def draw_landmarks(self, frame_bgr: np.ndarray, landmarks_px: List[np.ndarray]) -> np.ndarray:
        """Draw hand landmarks on the given BGR frame."""
        annotated = frame_bgr.copy()
        height, width = annotated.shape[:2]
        for hand_idx, coords in enumerate(landmarks_px):
            # Convert pixel coords back to normalized for MediaPipe drawing utils
            landmark_proto_list = []
            for x_px, y_px, z in coords:
                lm = self._mp_hands.NormalizedLandmark(x=float(x_px / width), y=float(y_px / height), z=float(z))
                landmark_proto_list.append(lm)
            landmark_set = self._mp_hands.NormalizedLandmarkList(landmark=landmark_proto_list)
            self._mp_drawing.draw_landmarks(
                annotated,
                landmark_set,
                self._mp_hands.HAND_CONNECTIONS,
                self._mp_styles.get_default_hand_landmarks_style(),
                self._mp_styles.get_default_hand_connections_style(),
            )
        return annotated


