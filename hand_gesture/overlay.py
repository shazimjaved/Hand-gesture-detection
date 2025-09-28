"""
Overlay module for drawing hand landmarks and gesture information on video frames.

This module provides functions to draw hand landmarks, gesture labels, and other
visual information on OpenCV video frames.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional


class VideoOverlay:
    """Handles drawing overlays on video frames."""
    
    def __init__(self, font_scale: float = 0.7, font_thickness: int = 2, theme: str = 'dark'):
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = font_scale
        self.font_thickness = font_thickness
        self.theme = theme if theme in ('dark', 'light') else 'dark'
        
        # Colors (BGR format)
        self._recompute_colors()

    def set_theme(self, theme: str) -> None:
        self.theme = theme if theme in ('dark', 'light') else 'dark'
        self._recompute_colors()

    def _recompute_colors(self) -> None:
        if self.theme == 'dark':
            self.colors = {
                'landmark': (0, 255, 0),
                'connection': (255, 0, 0),
                'gesture_label': (0, 215, 255),
                'confidence': (255, 255, 0),
                'background': (20, 20, 20),
                'panel': (40, 40, 40),
                'panel_border': (90, 90, 90),
                'text': (240, 240, 240),
                'bar_bg': (80, 80, 80),
                'bar_fg': (0, 200, 255),
            }
        else:
            self.colors = {
                'landmark': (0, 128, 0),
                'connection': (128, 0, 0),
                'gesture_label': (0, 150, 200),
                'confidence': (40, 40, 40),
                'background': (245, 245, 245),
                'panel': (255, 255, 255),
                'panel_border': (200, 200, 200),
                'text': (20, 20, 20),
                'bar_bg': (220, 220, 220),
                'bar_fg': (0, 140, 220),
            }
    
    def draw_landmarks(self, frame: np.ndarray, landmarks_list: List[np.ndarray]) -> np.ndarray:
        """
        Draw hand landmarks on the frame.
        
        Args:
            frame: Input video frame
            landmarks_list: List of landmark arrays for each detected hand
            
        Returns:
            Frame with landmarks drawn
        """
        annotated_frame = frame.copy()
        
        for landmarks in landmarks_list:
            if landmarks is None or len(landmarks) != 21:
                continue
                
            # Draw landmarks as circles
            for landmark in landmarks:
                x, y = int(landmark[0]), int(landmark[1])
                cv2.circle(annotated_frame, (x, y), 3, self.colors['landmark'], -1)
            
            # Draw connections between landmarks
            self._draw_hand_connections(annotated_frame, landmarks)
        
        return annotated_frame

    def draw_face_landmarks(self, frame: np.ndarray, landmarks_list: List[np.ndarray]) -> np.ndarray:
        """
        Draw face landmarks on the frame (similar to hand landmarks style).
        
        Args:
            frame: Input video frame
            landmarks_list: List of landmark arrays for each detected face
            
        Returns:
            Frame with face landmarks drawn
        """
        annotated_frame = frame.copy()
        
        for landmarks in landmarks_list:
            if landmarks is None or len(landmarks) not in [468, 478]:
                continue
                
            # Draw all landmarks as circles (like hand landmarks)
            for landmark in landmarks:
                x, y = int(landmark[0]), int(landmark[1])
                # Ensure coordinates are within frame bounds
                if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
                    cv2.circle(annotated_frame, (x, y), 3, self.colors['landmark'], -1)
            
            # Draw connections between landmarks (like hand connections)
            self._draw_face_connections(annotated_frame, landmarks)
        
        return annotated_frame
    
    def _draw_face_connections(self, frame: np.ndarray, landmarks: np.ndarray) -> None:
        """Draw connections between face landmarks (similar to hand connections)."""
        # Face connections based on MediaPipe face mesh
        # Using a subset of key connections for performance
        connections = [
            # Face outline
            (10, 338), (338, 297), (297, 332), (332, 284), (284, 251), (251, 389), (389, 356), (356, 454), (454, 323), (323, 361), (361, 288), (288, 397), (397, 365), (365, 379), (379, 378), (378, 400), (400, 377), (377, 152), (152, 148), (148, 176), (176, 149), (149, 150), (150, 136), (136, 172), (172, 58), (58, 132), (132, 93), (93, 234), (234, 127), (127, 162), (162, 21), (21, 54), (54, 103), (103, 67), (67, 109), (109, 10),
            # Left eyebrow
            (70, 63), (63, 105), (105, 66), (66, 107), (107, 55), (55, 65), (65, 52), (52, 53), (53, 46),
            # Right eyebrow  
            (296, 334), (334, 293), (293, 300), (300, 276), (276, 283), (283, 282), (282, 295), (295, 285), (285, 336),
            # Left eye
            (33, 7), (7, 163), (163, 144), (144, 145), (145, 153), (153, 154), (154, 155), (155, 133), (133, 173), (173, 157), (157, 158), (158, 159), (159, 160), (160, 161), (161, 246), (246, 33),
            # Right eye
            (362, 382), (382, 381), (381, 380), (380, 374), (374, 373), (373, 390), (390, 249), (249, 263), (263, 466), (466, 388), (388, 387), (387, 386), (386, 385), (385, 384), (384, 398), (398, 362),
            # Nose
            (1, 2), (2, 5), (5, 4), (4, 6), (6, 19), (19, 20), (20, 94), (94, 125), (125, 141), (141, 235), (235, 236), (236, 3), (3, 51), (51, 48), (48, 115), (115, 131), (131, 134), (134, 102), (102, 49), (49, 220), (220, 305), (305, 281), (281, 360), (360, 279), (279, 331), (331, 294), (294, 358), (358, 327), (327, 326), (326, 2),
            # Mouth outer
            (61, 84), (84, 17), (17, 314), (314, 405), (405, 320), (320, 307), (307, 375), (375, 321), (321, 308), (308, 324), (324, 318), (318, 61),
            # Mouth inner
            (78, 95), (95, 88), (88, 178), (178, 87), (87, 14), (14, 317), (317, 402), (402, 318), (318, 324), (324, 308), (308, 415), (415, 310), (310, 311), (311, 312), (312, 13), (13, 82), (82, 81), (81, 80), (80, 78)
        ]
        
        for start_idx, end_idx in connections:
            if start_idx < len(landmarks) and end_idx < len(landmarks):
                start_point = (int(landmarks[start_idx][0]), int(landmarks[start_idx][1]))
                end_point = (int(landmarks[end_idx][0]), int(landmarks[end_idx][1]))
                # Ensure coordinates are within frame bounds
                if (0 <= start_point[0] < frame.shape[1] and 0 <= start_point[1] < frame.shape[0] and
                    0 <= end_point[0] < frame.shape[1] and 0 <= end_point[1] < frame.shape[0]):
                    cv2.line(frame, start_point, end_point, self.colors['connection'], 2)
    
    def _draw_hand_connections(self, frame: np.ndarray, landmarks: np.ndarray) -> None:
        """Draw connections between hand landmarks."""
        # Hand connections based on MediaPipe hand model
        connections = [
            # Thumb
            (0, 1), (1, 2), (2, 3), (3, 4),
            # Index finger
            (0, 5), (5, 6), (6, 7), (7, 8),
            # Middle finger
            (0, 9), (9, 10), (10, 11), (11, 12),
            # Ring finger
            (0, 13), (13, 14), (14, 15), (15, 16),
            # Pinky
            (0, 17), (17, 18), (18, 19), (19, 20),
            # Palm
            (5, 9), (9, 13), (13, 17)
        ]
        
        for start_idx, end_idx in connections:
            start_point = (int(landmarks[start_idx][0]), int(landmarks[start_idx][1]))
            end_point = (int(landmarks[end_idx][0]), int(landmarks[end_idx][1]))
            cv2.line(frame, start_point, end_point, self.colors['connection'], 2)
    
    def draw_gesture_info(self, frame: np.ndarray, gesture_results: List[Tuple[str, float]], 
                         handedness_list: List[str], position: Tuple[int, int] = (20, 0)) -> np.ndarray:
        """
        Draw gesture information on the frame.
        
        Args:
            frame: Input video frame
            gesture_results: List of (gesture_name, confidence) tuples for hands
            handedness_list: List of handedness labels
            position: Position to start drawing text (x, y)
            
        Returns:
            Frame with gesture information drawn
        """
        annotated_frame = frame.copy()
        h, w = annotated_frame.shape[:2]

        # Place panel at bottom-left with margin
        margin = 20
        x = margin
        line_height = 26

        # Compose lines (limit to first 2 hands to avoid clutter)
        lines = ["Hand Gestures:"]
        for (gesture, confidence), handedness in list(zip(gesture_results, handedness_list))[:2]:
            lines.append(f"{handedness}: {gesture}  ({confidence:.2f})")

        # Measure panel width by longest line
        max_text_width = 0
        for line in lines:
            ((tw, th), _) = cv2.getTextSize(line, self.font, self.font_scale, self.font_thickness)
            if tw > max_text_width:
                max_text_width = tw

        panel_width = max(280, max_text_width + 30)
        panel_height = line_height * len(lines) + 20
        y_top = h - panel_height - margin
        y = y_top + 20

        # Translucent panel
        overlay_layer = annotated_frame.copy()
        cv2.rectangle(overlay_layer, (x - 10, y_top), (x - 10 + panel_width, y_top + panel_height), self.colors['panel'], -1)
        cv2.rectangle(overlay_layer, (x - 10, y_top), (x - 10 + panel_width, y_top + panel_height), self.colors['panel_border'], 2)
        alpha = 0.85
        annotated_frame = cv2.addWeighted(overlay_layer, alpha, annotated_frame, 1 - alpha, 0)

        # Draw lines
        for idx, line in enumerate(lines):
            if line == "":  # Skip empty lines
                y += line_height
                continue
            color = self.colors['text'] if line.endswith(":") else self.colors['gesture_label']
            cv2.putText(annotated_frame, line, (x, y), self.font, self.font_scale, color, self.font_thickness)
            y += line_height

        return annotated_frame
    
    def draw_instructions(self, frame: np.ndarray, position: Tuple[int, int] = (10, 10)) -> np.ndarray:
        """
        Draw control instructions on the frame.
        
        Args:
            frame: Input video frame
            position: Position to draw instructions (x, y)
            
        Returns:
            Frame with instructions drawn
        """
        annotated_frame = frame.copy()
        x, y = position
        
        instructions = [
            "Hand Gesture Detection:",
            "👍 Thumbs Up",
            "👎 Thumbs Down", 
            "☝ Index Pointing",
            "✌ Peace Sign",
            "✊ Fist",
            "🖐 Open Palm",
            "Press 'q' to quit"
        ]
        
        # Draw background
        text_height = len(instructions) * 25 + 10
        cv2.rectangle(annotated_frame, (x - 5, y - 5), (x + 400, y + text_height), 
                     self.colors['background'], -1)
        cv2.rectangle(annotated_frame, (x - 5, y - 5), (x + 400, y + text_height), 
                     self.colors['text'], 2)
        
        # Draw instructions
        for instruction in instructions:
            cv2.putText(annotated_frame, instruction, (x, y), 
                       self.font, self.font_scale * 0.6, self.colors['text'], self.font_thickness)
            y += 25
        
        return annotated_frame
    
    def draw_fps(self, frame: np.ndarray, fps: float, position: Tuple[int, int] = (10, 10)) -> np.ndarray:
        """
        Draw FPS counter on the frame.
        
        Args:
            frame: Input video frame
            fps: Current FPS value
            position: Position to draw FPS (x, y)
            
        Returns:
            Frame with FPS drawn
        """
        annotated_frame = frame.copy()
        x, y = position
        
        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(annotated_frame, fps_text, (x, y), 
                   self.font, self.font_scale, self.colors['text'], self.font_thickness)
        
        return annotated_frame
    
    def draw_hand_count(self, frame: np.ndarray, hand_count: int, 
                       position: Tuple[int, int] = (10, 40)) -> np.ndarray:
        """
        Draw hand count on the frame.
        
        Args:
            frame: Input video frame
            hand_count: Number of detected hands
            position: Position to draw count (x, y)
            
        Returns:
            Frame with hand count drawn
        """
        annotated_frame = frame.copy()
        x, y = position
        
        count_text = f"Hands Detected: {hand_count}"
        cv2.putText(annotated_frame, count_text, (x, y), 
                   self.font, self.font_scale, self.colors['text'], self.font_thickness)
        
        return annotated_frame



def create_overlay(frame: np.ndarray, landmarks_list: List[np.ndarray], 
                  gesture_results: List[Tuple[str, float]], handedness_list: List[str],
                  face_landmarks_list: List[np.ndarray] = None,
                  fps: Optional[float] = None, show_instructions: bool = False) -> np.ndarray:
    """
    Create a complete overlay with all visual elements.
    
    Args:
        frame: Input video frame
        landmarks_list: List of landmark arrays for each detected hand
        gesture_results: List of (gesture_name, confidence) tuples for hands
        handedness_list: List of handedness labels
        face_landmarks_list: List of face landmark arrays (optional)
        fps: Current FPS (optional)
        show_instructions: Whether to show control instructions
        
    Returns:
        Frame with complete overlay
    """
    overlay = VideoOverlay()
    
    # Draw hand landmarks
    annotated_frame = overlay.draw_landmarks(frame, landmarks_list)
    
    # Draw face landmarks
    if face_landmarks_list:
        annotated_frame = overlay.draw_face_landmarks(annotated_frame, face_landmarks_list)
    
    # Draw gesture information
    if gesture_results:
        annotated_frame = overlay.draw_gesture_info(
            annotated_frame, gesture_results, handedness_list
        )
    
    # Draw FPS counter
    if fps is not None:
        annotated_frame = overlay.draw_fps(annotated_frame, fps)
    
    # Draw hand count
    annotated_frame = overlay.draw_hand_count(annotated_frame, len(landmarks_list))
    
    return annotated_frame
