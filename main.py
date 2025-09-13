
import cv2
import time
import logging
from typing import Optional

from hand_gesture.capture import VideoCapture
from hand_gesture.detection import HandDetector
from hand_gesture.gestures import classify_hand_gestures
from hand_gesture.overlay import create_overlay
from hand_gesture.yolo_detect import YoloGestureDetector
from hand_gesture.face import FaceMeshDetector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HandGestureApp:
    """Main application class for hand gesture detection and control."""
    
    def __init__(self, camera_index: int = 0, frame_size: tuple = (640, 480)):
        self.camera_index = camera_index
        self.frame_size = frame_size
        self.running = False
        
        # Initialize components
        self.video_capture = VideoCapture(camera_index, frame_size)
        self.hand_detector = HandDetector(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
            use_yolo=True,
            yolo_model_path=None  # set to a hand model path to enable YOLO, else fallback
        )
        # Optional: pure YOLO gesture detector (end-to-end, no landmarks)
        self.yolo_gesture_detector = None  # YoloGestureDetector(weights_path) to enable
        # Face mesh detector
        self.face_detector = FaceMeshDetector(max_num_faces=1)
        
        # Performance tracking
        self.fps_counter = FPSCounter()
        
        logger.info("Hand Gesture App initialized")
    
    def run(self):
        """Run the main application loop."""
        try:
            self.video_capture.open()
            self.running = True
            
            logger.info("Starting hand gesture detection... Press 'q' to quit.")
            show_help = False
            show_overlay = True
            theme = 'dark'

            # Fullscreen window setup
            window_name = 'Hand Gesture Detection'
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            
            for success, frame in self.video_capture.frames():
                if not success:
                    logger.warning("Failed to read frame from camera")
                    continue
                
                # Process frame
                processed_frame = self._process_frame(frame, show_help, show_overlay, theme)
                
                # Display frame (fullscreen)
                cv2.imshow(window_name, processed_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                # Overlay/theme toggles disabled to keep UI clean
                
                # Update FPS
                self.fps_counter.update()
        
        except KeyboardInterrupt:
            logger.info("Application interrupted by user")
        except Exception as e:
            logger.error(f"Application error: {e}")
        finally:
            self.cleanup()
    
    def _process_frame(self, frame, show_help: bool = True, show_overlay: bool = True, theme: str = 'dark'):
        """Process a single video frame."""
        gesture_results = []
        landmarks_list = []
        handedness_list = []

        if self.yolo_gesture_detector is not None:
            # YOLO-only: get labels but don't draw boxes
            _annotated, detections = self.yolo_gesture_detector.predict(frame)
            gesture_results = [(label.replace('_', ' ').title(), conf) for label, conf in detections]
        else:
            # MediaPipe landmarks + classification
            frame_rgb, landmarks_list, handedness_list = self.hand_detector.process(frame)
            if landmarks_list:
                gesture_results = classify_hand_gestures(landmarks_list, handedness_list)

        # Face landmarks
        faces_list = self.face_detector.process(frame)

        # Overlay with FPS, gesture labels+confidence, hand and face landmarks
        processed_frame = create_overlay(
            frame,
            landmarks_list,
            gesture_results,
            handedness_list,
            fps=self.fps_counter.get_fps(),
            show_instructions=False,
            faces_list=faces_list,
        )
        
        return processed_frame
    
    # System actions have been removed for a simplified display-only experience.
    
    def cleanup(self):
        """Clean up resources."""
        self.running = False
        self.video_capture.release()
        cv2.destroyAllWindows()
        logger.info("Application cleaned up")


class FPSCounter:
    """Simple FPS counter for performance monitoring."""
    
    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        self.frame_times = []
        self.last_time = time.time()
    
    def update(self):
        """Update FPS counter with current frame."""
        current_time = time.time()
        frame_time = current_time - self.last_time
        self.last_time = current_time
        
        self.frame_times.append(frame_time)
        if len(self.frame_times) > self.window_size:
            self.frame_times.pop(0)
    
    def get_fps(self) -> float:
        """Get current FPS."""
        if not self.frame_times:
            return 0.0
        avg_frame_time = sum(self.frame_times) / len(self.frame_times)
        return 1.0 / avg_frame_time if avg_frame_time > 0 else 0.0


def main():
    """Main entry point for the application."""
    print("Hand Gesture Detection Application")
    print("=" * 40)
    print("Press 'q' to quit")
    print("=" * 40)
    
    try:
        app = HandGestureApp(camera_index="gesture.mp4", frame_size=(640, 480))
        app.run()
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        print(f"Error: {e}")
        print("Make sure your camera is connected and accessible.")


if __name__ == "__main__":
    main()
