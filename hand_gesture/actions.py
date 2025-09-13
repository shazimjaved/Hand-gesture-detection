"""
System control actions module for gesture-based interactions.

This module provides system control functions for volume, mouse, and media control
based on detected hand gestures.
"""

import time
from typing import Dict, Callable, Any, Optional
import logging

try:
    import pyautogui
    import pycaw
    from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
    from ctypes import cast, POINTER
    import comtypes
except ImportError as exc:
    raise ImportError(
        "Required packages not installed. Run: pip install pyautogui pycaw"
    ) from exc

# Configure pyautogui
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.1

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VolumeController:
    """Windows volume control using pycaw."""
    
    def __init__(self):
        self.volume_interface = None
        self._init_volume_control()
    
    def _init_volume_control(self):
        """Initialize Windows volume control interface."""
        try:
            devices = AudioUtilities.GetSpeakers()
            interface = devices.Activate(
                IAudioEndpointVolume._iid_, comtypes.CLSCTX_ALL, None
            )
            self.volume_interface = cast(interface, POINTER(IAudioEndpointVolume))
            logger.info("Volume control initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize volume control: {e}")
            self.volume_interface = None
    
    def get_volume(self) -> float:
        """Get current system volume (0.0 to 1.0)."""
        if self.volume_interface is None:
            return 0.0
        try:
            return self.volume_interface.GetMasterScalarVolume()
        except Exception as e:
            logger.error(f"Failed to get volume: {e}")
            return 0.0
    
    def set_volume(self, volume: float) -> bool:
        """Set system volume (0.0 to 1.0)."""
        if self.volume_interface is None:
            return False
        try:
            volume = max(0.0, min(1.0, volume))
            self.volume_interface.SetMasterScalarVolume(volume, None)
            return True
        except Exception as e:
            logger.error(f"Failed to set volume: {e}")
            return False
    
    def increase_volume(self, step: float = 0.1) -> bool:
        """Increase volume by step amount."""
        current_volume = self.get_volume()
        new_volume = min(1.0, current_volume + step)
        return self.set_volume(new_volume)
    
    def decrease_volume(self, step: float = 0.1) -> bool:
        """Decrease volume by step amount."""
        current_volume = self.get_volume()
        new_volume = max(0.0, current_volume - step)
        return self.set_volume(new_volume)


class MouseController:
    """Mouse control using pyautogui."""
    
    def __init__(self, screen_width: int = 1920, screen_height: int = 1080):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.last_position = None
        self.smoothing_factor = 0.3
    
    def move_cursor(self, landmarks, frame_width: int, frame_height: int) -> bool:
        """
        Move mouse cursor based on hand position.
        
        Args:
            landmarks: Hand landmarks array
            frame_width: Video frame width
            frame_height: Video frame height
            
        Returns:
            True if cursor was moved successfully
        """
        try:
            # Use wrist position for cursor control
            wrist = landmarks[0]  # WRIST landmark
            x_ratio = wrist[0] / frame_width
            y_ratio = wrist[1] / frame_height
            
            # Convert to screen coordinates
            screen_x = int(x_ratio * self.screen_width)
            screen_y = int(y_ratio * self.screen_height)
            
            # Apply smoothing
            if self.last_position:
                screen_x = int(self.last_position[0] * (1 - self.smoothing_factor) + 
                              screen_x * self.smoothing_factor)
                screen_y = int(self.last_position[1] * (1 - self.smoothing_factor) + 
                              screen_y * self.smoothing_factor)
            
            # Move cursor
            pyautogui.moveTo(screen_x, screen_y)
            self.last_position = (screen_x, screen_y)
            return True
            
        except Exception as e:
            logger.error(f"Failed to move cursor: {e}")
            return False


class MediaController:
    """Media control using pyautogui."""
    
    def __init__(self):
        self.last_action_time = 0
        self.action_cooldown = 1.0  # 1 second cooldown between actions
    
    def play_pause(self) -> bool:
        """Toggle play/pause for media."""
        current_time = time.time()
        if current_time - self.last_action_time < self.action_cooldown:
            return False
        
        try:
            pyautogui.press('space')
            self.last_action_time = current_time
            logger.info("Media play/pause toggled")
            return True
        except Exception as e:
            logger.error(f"Failed to toggle play/pause: {e}")
            return False


class GestureActionRegistry:
    """Registry for mapping gestures to actions."""
    
    def __init__(self):
        self.volume_controller = VolumeController()
        self.mouse_controller = MouseController()
        self.media_controller = MediaController()
        
        # Action registry
        self.actions: Dict[str, Callable] = {
            "Thumbs Up": self._thumbs_up_action,
            "Thumbs Down": self._thumbs_down_action,
            "Index Pointing": self._index_pointing_action,
            "Peace Sign": self._peace_sign_action,
            "Fist": self._fist_action,
            "Open Palm": self._open_palm_action,
        }
        
        # Action state tracking
        self.last_gesture = None
        self.gesture_start_time = None
        self.gesture_hold_duration = 0.5  # Minimum hold time for actions
    
    def execute_action(self, gesture: str, landmarks = None, 
                      frame_width: int = 640, frame_height: int = 480) -> bool:
        """
        Execute action for given gesture.
        
        Args:
            gesture: Detected gesture name
            landmarks: Hand landmarks (for cursor control)
            frame_width: Video frame width
            frame_height: Video frame height
            
        Returns:
            True if action was executed successfully
        """
        if gesture not in self.actions:
            return False
        
        # Track gesture duration
        current_time = time.time()
        if gesture != self.last_gesture:
            self.last_gesture = gesture
            self.gesture_start_time = current_time
            return False
        
        # Check if gesture has been held long enough
        if self.gesture_start_time and (current_time - self.gesture_start_time) < self.gesture_hold_duration:
            return False
        
        # Execute action
        try:
            return self.actions[gesture](landmarks, frame_width, frame_height)
        except Exception as e:
            logger.error(f"Failed to execute action for {gesture}: {e}")
            return False
    
    def _thumbs_up_action(self, landmarks, 
                         frame_width: int, frame_height: int) -> bool:
        """Increase system volume."""
        success = self.volume_controller.increase_volume(0.1)
        if success:
            logger.info("Volume increased")
        return success
    
    def _thumbs_down_action(self, landmarks, 
                           frame_width: int, frame_height: int) -> bool:
        """Decrease system volume."""
        success = self.volume_controller.decrease_volume(0.1)
        if success:
            logger.info("Volume decreased")
        return success
    
    def _index_pointing_action(self, landmarks, 
                              frame_width: int, frame_height: int) -> bool:
        """Control mouse cursor movement."""
        if landmarks is not None:
            return self.mouse_controller.move_cursor(landmarks, frame_width, frame_height)
        return False
    
    def _peace_sign_action(self, landmarks, 
                          frame_width: int, frame_height: int) -> bool:
        """Toggle media play/pause."""
        return self.media_controller.play_pause()
    
    def _fist_action(self, landmarks, 
                    frame_width: int, frame_height: int) -> bool:
        """Fist gesture - no action currently assigned."""
        logger.debug("Fist gesture detected - no action assigned")
        return False
    
    def _open_palm_action(self, landmarks, 
                         frame_width: int, frame_height: int) -> bool:
        """Open palm gesture - no action currently assigned."""
        logger.debug("Open palm gesture detected - no action assigned")
        return False
    
    def add_custom_action(self, gesture_name: str, action_func: Callable) -> None:
        """Add a custom gesture action."""
        self.actions[gesture_name] = action_func
        logger.info(f"Added custom action for gesture: {gesture_name}")
    
    def get_available_gestures(self) -> list:
        """Get list of available gesture names."""
        return list(self.actions.keys())


# Global action registry instance
action_registry = GestureActionRegistry()
