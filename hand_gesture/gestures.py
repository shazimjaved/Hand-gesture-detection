"""
Gesture classification module for hand gesture recognition.

This module provides gesture classification based on MediaPipe hand landmarks.
Supports: fist, open palm, thumbs up, thumbs down, index pointing, peace sign, rock.
Relaxed rules for more comfortable gesture detection.
"""

from typing import List, Tuple
import numpy as np


class GestureClassifier:
    """Classifies hand gestures based on MediaPipe landmarks with relaxed rules."""

    # MediaPipe hand landmark indices
    THUMB_TIP = 4
    THUMB_IP = 3
    THUMB_MCP = 2
    INDEX_TIP = 8
    INDEX_PIP = 6
    INDEX_MCP = 5
    MIDDLE_TIP = 12
    MIDDLE_PIP = 10
    MIDDLE_MCP = 9
    RING_TIP = 16
    RING_PIP = 14
    RING_MCP = 13
    PINKY_TIP = 20
    PINKY_PIP = 18
    PINKY_MCP = 17
    WRIST = 0

    def __init__(self, confidence_threshold: float = 0.5):
        self.confidence_threshold = confidence_threshold
        # Very relaxed tolerance factors for better detection
        self.finger_tolerance = 0.04  # Increased for more lenient finger detection
        self.thumb_clearance = 0.02  # Decreased for easier thumb detection
        self.fist_fold_threshold = 0.5  # Much more lenient for fist detection
        self.min_fingers_folded = 2  # Only need 2 fingers folded for fist

    def classify_gesture(self, landmarks: np.ndarray, handedness: str = "Right") -> Tuple[str, float]:
        """
        Classify gesture from hand landmarks with relaxed rules.

        Args:
            landmarks: Array of shape (21, 3) with x, y, z coordinates
            handedness: "Left" or "Right" hand

        Returns:
            Tuple of (gesture_name, confidence_score)
        """
        if landmarks is None or len(landmarks) != 21:
            return "Unknown", 0.0

        # Determine finger states (per finger) with relaxed thresholds
        index_up = self._is_finger_up(landmarks, self.INDEX_TIP, self.INDEX_PIP, self.INDEX_MCP, handedness)
        middle_up = self._is_finger_up(landmarks, self.MIDDLE_TIP, self.MIDDLE_PIP, self.MIDDLE_MCP, handedness)
        ring_up = self._is_finger_up(landmarks, self.RING_TIP, self.RING_PIP, self.RING_MCP, handedness)
        pinky_up = self._is_finger_up(landmarks, self.PINKY_TIP, self.PINKY_PIP, self.PINKY_MCP, handedness)
        non_thumb_up = sum([index_up, middle_up, ring_up, pinky_up])
        thumb_is_up = self._is_thumb_up(landmarks, handedness)
        thumb_is_down = self._is_thumb_down(landmarks, handedness)

        # --- RELAXED GESTURE DETECTION (ORDER MATTERS) ---

        # Open palm: Most specific - 3+ fingers up with thumb up
        if non_thumb_up >= 3 and thumb_is_up:
            return "Open Palm", 0.85

        # Index pointing: Very specific - only index up, others down
        if index_up and not middle_up and not ring_up and not pinky_up:
            return "Index Pointing", 0.80

        # Thumbs up: Very specific - thumb up with NO other fingers up
        if thumb_is_up and non_thumb_up == 0:
            return "Thumbs Up", 0.80

        # Thumbs down: Very specific - thumb down with NO other fingers up
        if thumb_is_down and non_thumb_up == 0:
            return "Thumbs Down", 0.80

        # Rock: Specific pattern - index and pinky up, middle and ring down
        if index_up and pinky_up and not middle_up and not ring_up:
            return "Rock", 0.75
        elif index_up and pinky_up and (middle_up or ring_up) and non_thumb_up <= 3:
            return "Rock", 0.70

        # Peace sign: Specific pattern - index and middle up, others down
        if index_up and middle_up and not ring_up and not pinky_up:
            return "Peace Sign", 0.80
        elif index_up and middle_up and (ring_up or pinky_up) and non_thumb_up <= 3:
            return "Peace Sign", 0.70

        # Fist: Multiple detection methods for better reliability
        if self._is_fist(landmarks):
            return "Fist", 0.85
        
        # Alternative fist detection: if most fingers are down
        if non_thumb_up <= 1:
            return "Fist", 0.80

        # Partial fist: If some fingers are down but not a complete fist
        if non_thumb_up <= 2 and not self._is_fist(landmarks):
            return "Partial Fist", 0.65

        return "Unknown", 0.30

    # ----------------- Helpers -----------------

    def _is_thumb_up(self, landmarks: np.ndarray, handedness: str) -> bool:
        """Thumb is up if tip above IP and sticking out sideways (relaxed)."""
        thumb_tip = landmarks[self.THUMB_TIP]
        thumb_ip = landmarks[self.THUMB_IP]
        index_mcp = landmarks[self.INDEX_MCP]

        # condition 1: vertical check (tip above IP in image coords) - more lenient
        vertical_up = thumb_tip[1] < thumb_ip[1] + 0.01  # Allow small margin

        # condition 2: sideways clearance (thumb sticks out from palm) - more lenient
        side_clearance = abs(thumb_tip[0] - index_mcp[0]) > self.thumb_clearance

        return bool(vertical_up and side_clearance)

    def _is_thumb_down(self, landmarks: np.ndarray, handedness: str) -> bool:
        """Thumb is down if tip below IP joint in image coordinates (y larger) - relaxed."""
        thumb_tip = landmarks[self.THUMB_TIP]
        thumb_ip = landmarks[self.THUMB_IP]
        # More lenient - allow small margin
        return bool(thumb_tip[1] > thumb_ip[1] - 0.01)

    def _is_finger_up(self, landmarks: np.ndarray, tip_idx: int, pip_idx: int, mcp_idx: int, handedness: str) -> bool:
        """Check if a specific finger is extended with relaxed margin based on hand size."""
        tip = landmarks[tip_idx]
        pip = landmarks[pip_idx]
        size = self._hand_size(landmarks)
        margin = self.finger_tolerance * size  # Use configurable tolerance
        return (pip[1] - tip[1]) > margin

    def _hand_size(self, landmarks: np.ndarray) -> float:
        """Rough hand size: wrist to middle MCP distance."""
        wrist = landmarks[self.WRIST][:2]
        middle_mcp = landmarks[self.MIDDLE_MCP][:2]
        return float(np.linalg.norm(wrist - middle_mcp) + 1e-6)

    def _is_fist(self, landmarks: np.ndarray) -> bool:
        """Detect fist by checking if most fingers are folded (very lenient)."""
        size = self._hand_size(landmarks)
        dist_thresh = self.fist_fold_threshold * size  # Use configurable threshold
        pip_margin = 0.06 * size   # Much more lenient tolerance

        finger_defs = [
            (self.INDEX_TIP, self.INDEX_PIP, self.INDEX_MCP),
            (self.MIDDLE_TIP, self.MIDDLE_PIP, self.MIDDLE_MCP),
            (self.RING_TIP, self.RING_PIP, self.RING_MCP),
            (self.PINKY_TIP, self.PINKY_PIP, self.PINKY_MCP),
        ]

        folded_count = 0
        for tip_idx, pip_idx, mcp_idx in finger_defs:
            tip_xy = landmarks[tip_idx][:2]
            mcp_xy = landmarks[mcp_idx][:2]
            pip_y = landmarks[pip_idx][1]
            tip_y = landmarks[tip_idx][1]
            dist_tip_mcp = np.linalg.norm(tip_xy - mcp_xy)

            # Very lenient conditions for fist detection
            # Check if finger is folded - either close to MCP OR tip below PIP OR very close to palm
            is_folded = (
                (dist_tip_mcp < dist_thresh) or 
                (tip_y >= pip_y - pip_margin) or
                (dist_tip_mcp < 0.3 * size)  # Very close to palm
            )
            
            if is_folded:
                folded_count += 1

        # Only need 2+ fingers folded = fist (very lenient)
        return folded_count >= self.min_fingers_folded


# ----------------- Multi-hand wrapper -----------------

def classify_hand_gestures(landmarks_list: List[np.ndarray], handedness_list: List[str]) -> List[Tuple[str, float]]:
    """
    Classify gestures for multiple hands.

    Args:
        landmarks_list: List of landmark arrays for each detected hand
        handedness_list: List of handedness labels for each hand

    Returns:
        List of (gesture_name, confidence) tuples for each hand
    """
    classifier = GestureClassifier()
    results = []

    for landmarks, handedness in zip(landmarks_list, handedness_list):
        gesture, confidence = classifier.classify_gesture(landmarks, handedness)
        results.append((gesture, confidence))

    return results
