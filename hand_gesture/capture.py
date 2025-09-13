import cv2
from typing import Generator, Optional, Tuple


class VideoCapture:
    """OpenCV video capture wrapper with safe initialization and cleanup."""

    def __init__(self, camera_index: int = 0, frame_size: Optional[Tuple[int, int]] = None):
        self.camera_index = camera_index
        self.frame_size = frame_size
        self.cap: Optional[cv2.VideoCapture] = None

    def open(self) -> None:
    # Agar int hua to webcam open karo (CAP_DSHOW ke sath)
        if isinstance(self.camera_index, int):
            self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        else:
        # Agar string hua to video file open karo (normal)
            self.cap = cv2.VideoCapture(self.camera_index)

        if self.frame_size is not None and isinstance(self.camera_index, int):
        # Sirf webcam pe frame size set karna
            width, height = self.frame_size
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        if not self.cap.isOpened():
            raise RuntimeError(f"Unable to open video source: {self.camera_index}")


    def frames(self) -> Generator[Tuple[bool, Optional[any]], None, None]:
        if self.cap is None:
            self.open()
        assert self.cap is not None
        while True:
            success, frame = self.cap.read()
            if not success:
                break   
            yield True, frame


    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None


