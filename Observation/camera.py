import cv2
from numpy import ndarray


class Camera:
    def __init__(self, cam_id: int):
        """Initialize the camera with the specified id."""
        self.cam_id = cam_id
        self.cap = cv2.VideoCapture(cam_id)
        if not self.cap.isOpened():
            raise IOError(f"Camera with id {cam_id} could not be opened.")

    def __del__(self):
        """Release the camera that is being used."""
        self.cap.release()

    def set_resolution(self, w: float, h: float):
        """Sets the resolution of the camera."""
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

    def read_frame(self) -> ndarray:
        """Reads a single frame from from the camera."""
        result, image = self.cap.read()
        if not result:
            raise IOError(f"Could not capture a frame from the camera with id {self.cam_id}")
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # image /= 255.0
        return image
