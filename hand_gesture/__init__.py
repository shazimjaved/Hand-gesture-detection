"""
Hand Gesture Detection Package

A modular Python package for real-time hand gesture detection and control
using OpenCV and MediaPipe.

Modules:
    capture: Video capture using OpenCV
    detection: Hand detection using MediaPipe
    gestures: Gesture classification algorithms
    actions: System control actions (volume, mouse, media)
    overlay: Video overlay and visualization
"""

__version__ = "1.0.0"
__author__ = "Hand Gesture Detection Team"

from .capture import VideoCapture
from .detection import HandDetector
from .gestures import GestureClassifier, classify_hand_gestures
from .actions import action_registry, VolumeController, MouseController, MediaController
from .overlay import VideoOverlay, create_overlay

__all__ = [
    'VideoCapture',
    'HandDetector', 
    'GestureClassifier',
    'classify_hand_gestures',
    'action_registry',
    'VolumeController',
    'MouseController', 
    'MediaController',
    'VideoOverlay',
    'create_overlay'
]
