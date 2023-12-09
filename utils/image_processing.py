import cv2 as cv
import numpy as np


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
            x, y, w, h = faces[0]
            return img[y: y + h, x+int(w*0.1): x + int(w*0.9)]
        else:
            return None

    def preprocess_face(self, face_img: np.ndarray):
        """
        Preprocesses the given face image.

        Args:
            face_img: An image array representing the face.

        Returns:
            np.ndarray: The preprocessed image of the face.
        """
        if face_img is not None:
            if len(face_img.shape) == 3:
                # look at only the red color
                # print(np.array(face_img).shape)
                gray_face = face_img[:, :, 2]
                # gray_face = cv.cvtColor(face_img, cv.COLOR_BGR2GRAY)
            else:
                gray_face = face_img

            amplified_face = cv.equalizeHist(gray_face)
            return amplified_face
        return None

    def get_face_frame_bytes(self, frame: np.ndarray, encoding: str = ".png"):
        """
        Processes a given frame to detect the face, preprocesses it and encodes it to bytes.

        Args:
            frame: The image frame.
            encoding: The image encoding format to be used for conversion.

        Returns:
            tuple: (face image bytes, face image numpy array).
        """
        face = self.get_face_area(frame)
        preprocessed_face = self.preprocess_face(face)

        if preprocessed_face is None:
            a = cv.imencode(encoding, np.full(
                (100, 100, 3), fill_value=100))[1]
            return a.tobytes(), a
        else:
            a = cv.imencode(encoding, preprocessed_face)[1]
            return a.tobytes(), a
