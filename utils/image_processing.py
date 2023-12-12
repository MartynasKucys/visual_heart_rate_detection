import cv2 as cv
import numpy as np
from copy import deepcopy as copy
from utils.magnification import Euler_Video_Magnification


class ImageProcessor:
    """
    This class implements detection and processing of faces.
    """

    def __init__(self, frame_rate):
        """
        Initializes haar-cascade face detection.
        """
        self.magnification = Euler_Video_Magnification(
            level=5, amplification=20, fps=frame_rate, backward_frames=30)
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
        # print(img.shape)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        if len(faces) >= 1:
            return faces[0]
        else:
            return None, None, None, None

    def preprocess_face(self, face_img: np.ndarray, EVM: Euler_Video_Magnification):
        """
        Preprocesses the given face image.

        Args:
            face_img: An image array representing the face.

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

    # def get_face_frame_bytes(self, frame: np.ndarray, EVM: Euler_Video_Magnification, encoding: str = ".png", ):
    #     """
    #     Processes a given frame to detect the face, preprocesses it and encodes it to bytes.

    #     Args:
    #         frame: The image frame.
    #         encoding: The image encoding format to be used for conversion.

    #     Returns:
    #         tuple: (face image bytes, face image numpy array).
    #     """
    #     x, y, h, w = self.get_face_area(frame)
    #     face = frame[y:y+w, x:x+h]
    #     # print("face", face.shape)

    #     preprocessed_face = self.preprocess_face(face, EVM)

    #     if preprocessed_face is None:
    #         a = cv.imencode(encoding, np.full(
    #             (100, 100, 3), fill_value=100))[1]
    #         return a.tobytes(), a
    #     else:
    #         a = cv.imencode(encoding, preprocessed_face)[1]
    #         return a.tobytes(), a
