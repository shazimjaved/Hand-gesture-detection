"""
Gesture classification module for hand gesture recognition.

This module provides gesture classification based on MediaPipe hand landmarks.
Supports: fist, open palm, thumbs up, thumbs down, index pointing, peace sign.
"""

from typing import List, Optional, Tuple
import numpy as np


class GestureClassifier:
    """Classifies hand gestures based on MediaPipe landmarks."""
    
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
    
    def __init__(self, confidence_threshold: float = 0.7):
        self.confidence_threshold = confidence_threshold
    
    def classify_gesture(self, landmarks: np.ndarray, handedness: str = "Right") -> Tuple[str, float]:
        """
        Classify gesture from hand landmarks.
        
        Args:
            landmarks: Array of shape (21, 3) with x, y, z coordinates
            handedness: "Left" or "Right" hand
            
        Returns:
            Tuple of (gesture_name, confidence_score)
        """
        if landmarks is None or len(landmarks) != 21:
            return "Unknown", 0.0
        
        # Determine finger states (per finger)
        index_up = self._is_finger_up(landmarks, self.INDEX_TIP, self.INDEX_PIP, self.INDEX_MCP, handedness)
        middle_up = self._is_finger_up(landmarks, self.MIDDLE_TIP, self.MIDDLE_PIP, self.MIDDLE_MCP, handedness)
        ring_up = self._is_finger_up(landmarks, self.RING_TIP, self.RING_PIP, self.RING_MCP, handedness)
        pinky_up = self._is_finger_up(landmarks, self.PINKY_TIP, self.PINKY_PIP, self.PINKY_MCP, handedness)
        non_thumb_up = sum([index_up, middle_up, ring_up, pinky_up])
        thumb_is_up = self._is_thumb_up(landmarks, handedness)
        thumb_is_down = self._is_thumb_down(landmarks, handedness)

        # Ordering matters to disambiguate similar poses
        # Open palm
        if non_thumb_up == 4 and thumb_is_up:
            return "Open Palm", 0.95

        # Thumbs up (others down)
        # Thumbs up (others mostly down)
        if thumb_is_up and non_thumb_up <= 1:
            return "Thumbs Up", 0.90


        # Thumbs down (others down) - check BEFORE fist
        if thumb_is_down and non_thumb_up == 0:
            return "Thumbs Down", 0.92

        # Rock: index and pinky up, middle and ring down (thumb any)
        if index_up and pinky_up and not middle_up and not ring_up:
            return "Rock", 0.88

        # Index pointing: index up, others down
        if index_up and not middle_up and not ring_up and not pinky_up:
            return "Index Pointing", 0.86

        # Peace sign: index and middle up, ring and pinky down
        if index_up and middle_up and not ring_up and not pinky_up:
            return "Peace Sign", 0.86

        # Fist: all non-thumb fingertips folded towards MCPs
        if self._is_fist(landmarks) and not thumb_is_up:
            return "Fist", 0.95

        return "Unknown", 0.35
    
    def _is_thumb_up(self, landmarks: np.ndarray, handedness: str) -> bool:
        thumb_tip = landmarks[self.THUMB_TIP]
        thumb_ip = landmarks[self.THUMB_IP]
        index_mcp = landmarks[self.INDEX_MCP]

    # condition 1: vertical check
        vertical_up = thumb_tip[1] < thumb_ip[1]

    # condition 2: sideways check (thumb should not overlap index base too much)
        side_clearance = abs(thumb_tip[0] - index_mcp[0]) > 0.02  # margin based on normalized x

        return bool(vertical_up and side_clearance)


    def _is_thumb_down(self, landmarks: np.ndarray, handedness: str) -> bool:
        """Thumb is down if thumb tip is below IP joint in image coordinates (y larger)."""
        thumb_tip = landmarks[self.THUMB_TIP]
        thumb_ip = landmarks[self.THUMB_IP]
        return bool(thumb_tip[1] > thumb_ip[1])
    
    
    def _count_non_thumb_fingers_up(self, landmarks: np.ndarray, handedness: str) -> int:
        """Count number of extended non-thumb fingers (index, middle, ring, pinky)."""
        fingers_up = 0
        if self._is_finger_up(landmarks, self.INDEX_TIP, self.INDEX_PIP, self.INDEX_MCP, handedness):
            fingers_up += 1
        if self._is_finger_up(landmarks, self.MIDDLE_TIP, self.MIDDLE_PIP, self.MIDDLE_MCP, handedness):
            fingers_up += 1
        if self._is_finger_up(landmarks, self.RING_TIP, self.RING_PIP, self.RING_MCP, handedness):
            fingers_up += 1
        if self._is_finger_up(landmarks, self.PINKY_TIP, self.PINKY_PIP, self.PINKY_MCP, handedness):
            fingers_up += 1
        return fingers_up
    
    def _is_finger_up(self, landmarks: np.ndarray, tip_idx: int, pip_idx: int, mcp_idx: int, handedness: str) -> bool:
        """Check if a specific finger is extended with margin based on hand size."""
        tip = landmarks[tip_idx]
        pip = landmarks[pip_idx]
        size = self._hand_size(landmarks)
        margin = 0.02 * size
        return (pip[1] - tip[1]) > margin
    
    def _is_index_pointing(self, landmarks: np.ndarray) -> bool:
        """Check if index finger is pointing (extended while others are down)."""
        # Index finger up
        index_up = self._is_finger_up(landmarks, self.INDEX_TIP, self.INDEX_PIP, self.INDEX_MCP, "Right")
        
        # Other fingers down
        middle_down = not self._is_finger_up(landmarks, self.MIDDLE_TIP, self.MIDDLE_PIP, self.MIDDLE_MCP, "Right")
        ring_down = not self._is_finger_up(landmarks, self.RING_TIP, self.RING_PIP, self.RING_MCP, "Right")
        pinky_down = not self._is_finger_up(landmarks, self.PINKY_TIP, self.PINKY_PIP, self.PINKY_MCP, "Right")
        
        return index_up and middle_down and ring_down and pinky_down
    
    def _is_peace_sign(self, landmarks: np.ndarray) -> bool:
        """Check if hand is making peace sign (index and middle fingers up)."""
        # Index and middle fingers up
        index_up = self._is_finger_up(landmarks, self.INDEX_TIP, self.INDEX_PIP, self.INDEX_MCP, "Right")
        middle_up = self._is_finger_up(landmarks, self.MIDDLE_TIP, self.MIDDLE_PIP, self.MIDDLE_MCP, "Right")
        
        # Ring and pinky fingers down
        ring_down = not self._is_finger_up(landmarks, self.RING_TIP, self.RING_PIP, self.RING_MCP, "Right")
        pinky_down = not self._is_finger_up(landmarks, self.PINKY_TIP, self.PINKY_PIP, self.PINKY_MCP, "Right")
        
        return index_up and middle_up and ring_down and pinky_down

    def _hand_size(self, landmarks: np.ndarray) -> float:
        """Rough hand size: wrist to middle MCP distance."""
        wrist = landmarks[self.WRIST][:2]
        middle_mcp = landmarks[self.MIDDLE_MCP][:2]
        return float(np.linalg.norm(wrist - middle_mcp) + 1e-6)

    def _is_fist(self, landmarks: np.ndarray) -> bool:
        size = self._hand_size(landmarks)
        dist_thresh = 0.8 * size   # pehle 0.65 tha, ab thoda relax kiya
        pip_margin = 0.04 * size   # thoda aur tolerance diya

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

        # Relaxed condition: allow small error
            if dist_tip_mcp < dist_thresh and tip_y >= pip_y - pip_margin:
                folded_count += 1

    # At least 3 fingers folded = fist (instead of exactly 4)
        return folded_count >= 3



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
