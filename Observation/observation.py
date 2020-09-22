from Observation.camera import Camera


class Observation:
    """Capture images from video streams."""

    def __init__(self, res_w: float, res_h: float):
        """Initializes the Observation module.

        Parameters:
            res_w = resolution (width)
            res_h = resolution (height)
        """
        self.res_h = res_h
        self.res_w = res_w
        self.stream = None

    def start_webcam(self, cam_id=0):
        """Enables a webcam as the source of the video stream.

        The webcam cannot be used by other applications until stop() is called."""
        self.stream = Camera(cam_id)
        self.stream.set_resolution(self.res_w, self.res_h)

    def stop(self):
        """Stops the currently active video stream."""
        if isinstance(self.stream, Camera):
            self.stream = None

    def read_frame(self):
        """Record a single image from the selected video device.

        Returns a three dimensional numpy array with the format (res_h, res_w, 3)."""
        if isinstance(self.stream, Camera):
            return self.stream.read_frame()
        else:
            raise Exception("read_frame() was called but there is no camera initialized.")
