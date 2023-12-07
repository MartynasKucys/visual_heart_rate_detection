import cv2 as cv


class Camera:
    """
    This class implements camera handling operations.
    """

    def __init__(self):
        """
        Initializes the camera instance by attempting to open a video capture device or a video file.
        """
        # self.cap = cv.VideoCapture(0)
        self.cap = cv.VideoCapture("video.mp4")

        if not self.cap.isOpened():
            raise Exception("Cannot open camera")

        self.frame_rate = int(self.cap.get(cv.CAP_PROP_FPS)) | 30

    def get_frame(self, encoding: str = ".png"):
        """
        Captures a single frame from the camera or video file.

        Args:
            encoding: The image encoding format to be used for conversion.

        Returns:
            tuple: (encoded frame bytes, original frame).
        """
        ret_val, frame = self.cap.read()
        if not ret_val:
            raise Exception("Failed to capture frame")
        img_bytes = cv.imencode(encoding, frame)[1].tobytes()

        return img_bytes, frame

    def release(self):
        """
        Releases the camera or video file resources.
        """
        self.cap.release()
        cv.destroyAllWindows()
