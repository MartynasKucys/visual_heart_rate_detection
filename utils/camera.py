import cv2 as cv


class Camera:
    def __init__(self):
        pass

    # def start(self):
    #     self.cap = cv.VideoCapture(0)
    #     if not self.cap.isOpened():
    #         raise Exception("Cannot open camera")
    #     return self.cap

    # Add the following lines in your code

    def start_capturing(self):
        self.cap = cv.VideoCapture(0)
        if not self.cap.isOpened():
            raise Exception("Cannot open camera")

        self.fps = self.cap.get(cv.CAP_PROP_FPS)  # Get frames per second
        if self.fps == 0.0:
            # If the camera doesn't provide fps information, you might need to set it manually
            self.fps = 30.0  # Set a default value (e.g., 30 fps)

        return self.cap, self.fps

    def end(self, cap):
        self.cap.release()
        cv.destroyAllWindows()
