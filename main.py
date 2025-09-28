import cv2
import time
import logging
from typing import Optional

from hand_gesture.capture import VideoCapture
from hand_gesture.detection import HandDetector
from hand_gesture.gestures import classify_hand_gestures
from hand_gesture.overlay import create_overlay
from hand_gesture.face_detection import FaceDetector

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
            use_yolo=False,
            yolo_model_path=None
        )
        self.face_detector = FaceDetector(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Video recording
        self.video_writer = None
        self.output_filename = "output.mp4"
        
        # Performance tracking
        self.fps_counter = FPSCounter()
        
        logger.info("Hand Gesture App initialized")
    
    def run(self):
        """Run the main application loop."""
        try:
            self.video_capture.open()
            self.running = True
            
            # Initialize video writer
            self._init_video_writer()
            
            logger.info("Starting hand gesture detection... Press 'q' to quit.")
            logger.info(f"Recording video to: {self.output_filename}")
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
                
                # Write frame to video file
                if self.video_writer is not None:
                    self.video_writer.write(processed_frame)
                
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
        face_landmarks_list = []

        # Hand detection and classification
        frame_rgb, landmarks_list, handedness_list = self.hand_detector.process(frame)
        if landmarks_list:
            gesture_results = classify_hand_gestures(landmarks_list, handedness_list)

        # Face detection
        _, face_landmarks_list = self.face_detector.process(frame)

        # Overlay with FPS, gesture labels+confidence, and landmarks
        processed_frame = create_overlay(
            frame,
            landmarks_list,
            gesture_results,
            handedness_list,
            face_landmarks_list,
            fps=self.fps_counter.get_fps(),
            show_instructions=False,
        )
        
        return processed_frame
    
    # System actions have been removed for a simplified display-only experience.
    
    def _init_video_writer(self):
        """Initialize video writer for recording."""
        try:
            # Define codec and create VideoWriter object
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = 30.0  # Frames per second
            self.video_writer = cv2.VideoWriter(
                self.output_filename, 
                fourcc, 
                fps, 
                self.frame_size
            )
            
            if not self.video_writer.isOpened():
                logger.error("Failed to initialize video writer")
                self.video_writer = None
            else:
                logger.info(f"Video writer initialized: {self.output_filename}")
                
        except Exception as e:
            logger.error(f"Error initializing video writer: {e}")
            self.video_writer = None

    def cleanup(self):
        """Clean up resources."""
        self.running = False
        
        # Release video writer
        if self.video_writer is not None:
            self.video_writer.release()
            logger.info(f"Video saved as: {self.output_filename}")
        
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
    print("=" * 50)
    print("Features:")
    print("- Hand Gestures: Thumbs Up, Peace Sign, Fist, etc.")
    print("Press 'q' to quit")
    print("=" * 50)
    
    try:
        app = HandGestureApp(camera_index=0, frame_size=(640, 480))
        app.run()
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        print(f"Error: {e}")
        print("Make sure your camera is connected and accessible.")


if __name__ == "__main__":
    main()
