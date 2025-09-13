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

    def draw_face_landmarks(self, frame: np.ndarray, faces_list: List[np.ndarray]) -> np.ndarray:
        """Draw facial landmarks as small dots."""
        annotated = frame.copy()
        for face in faces_list:
            for pt in face:
                cv2.circle(annotated, (int(pt[0]), int(pt[1])), 1, (255, 200, 0), -1)
        return annotated
    
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
            gesture_results: List of (gesture_name, confidence) tuples
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
            color = self.colors['text'] if idx == 0 else self.colors['gesture_label']
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
            "Controls:",
            "👍 Thumbs Up: Increase Volume",
            "👎 Thumbs Down: Decrease Volume", 
            "☝ Index Point: Move Mouse",
            "✌ Peace Sign: Play/Pause Media",
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
                  fps: Optional[float] = None, show_instructions: bool = False,
                  faces_list: Optional[List[np.ndarray]] = None) -> np.ndarray:
    """
    Create a complete overlay with all visual elements.
    
    Args:
        frame: Input video frame
        landmarks_list: List of landmark arrays for each detected hand
        gesture_results: List of (gesture_name, confidence) tuples
        handedness_list: List of handedness labels
        fps: Current FPS (optional)
        show_instructions: Whether to show control instructions
        
    Returns:
        Frame with complete overlay
    """
    overlay = VideoOverlay()
    
    # Draw landmarks
    annotated_frame = overlay.draw_landmarks(frame, landmarks_list)
    
    # Draw gesture information
    if gesture_results:
        annotated_frame = overlay.draw_gesture_info(annotated_frame, gesture_results, handedness_list)
    
    # Draw FPS counter
    if fps is not None:
        annotated_frame = overlay.draw_fps(annotated_frame, fps)
    
    # Draw hand count
    annotated_frame = overlay.draw_hand_count(annotated_frame, len(landmarks_list))
    
    # Draw face landmarks if provided
    if faces_list:
        annotated_frame = overlay.draw_face_landmarks(annotated_frame, faces_list)
    
    return annotated_frame
