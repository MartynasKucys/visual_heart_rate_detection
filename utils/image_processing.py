import cv2 as cv
import numpy as np
from copy import deepcopy as copy
from utils.magnification import EulerMagnification


class ImageProcessor:
    """
    This class implements detection and processing of faces.
    """

    def __init__(self):
        """
        Initializes haar-cascade face detection.
        """
        self.face_cascade = cv.CascadeClassifier(
            cv.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

    def get_face_area(self, img: np.ndarray):
        """
        Detects the first face in the given image using the Haar-cascade classifier and extracts its area.

        Args:
            img: The image (possibly) containing a face.

        Returns:
            np.ndarray | None: The cropped area of the image that contains the detected face. None if no face is detected.
        """

        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        if len(faces) >= 1:
            return faces[0]
        else:
            return None, None, None, None

    def preprocess_face(self, face_img: np.ndarray, EVM: EulerMagnification):
        """
        Preprocesses the given face image.

        Args:
            face_img: An image array representing the face.
            EVM: EulerMagnification instance

        Returns:
            np.ndarray: The preprocessed image of the face.
        """
        if face_img is not None:
            conv_img = face_img
            EVM.frames.append(copy(conv_img))
            EVM.apply_gaussian_pyramid(conv_img)
            if len(EVM.frames) > 20:
                magnified = EVM.magnify_color(conv_img, 0.4, 3)
                return magnified
            else:
                return face_img
        return None
