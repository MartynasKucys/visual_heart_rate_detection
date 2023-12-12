from utils.camera import Camera
from utils.image_processing import ImageProcessor
from utils.gui import GUI
from utils.calculations import HRSignalProcessor
from utils.magnification import EulerMagnification
import numpy as np
import PySimpleGUI as sg
import cv2 as cv


def run():
    """
    Runs the heart rate calculation application.
    """
    camera = Camera()
    image_processor = ImageProcessor()
    hr_processor = HRSignalProcessor(camera.frame_rate)
    gui = GUI(camera.frame_rate, time_window=5)
    EVM = EulerMagnification(level=3, amplification=20, fps=4)
    history_measurements = []

    while True:
        event, values = gui.read(timeout=1000 // camera.frame_rate)

        if event == "Exit" or event == sg.WIN_CLOSED:
            break

        current_frame_bytes, current_frame = camera.get_frame()
        x, y, h, w = image_processor.get_face_area(current_frame)

        amplified_face = image_processor.preprocess_face(current_frame, EVM)
        face = cv.convertScaleAbs(
            amplified_face[y : y + w, x : x + h]
            if x != None
            else np.zeros((100, 100, 3))
        )

        if amplified_face is not None and len(EVM.frames) >= 20:
            history_measurements.append(np.average(face))

        heart_rate = None
        if len(history_measurements) >= camera.frame_rate:
            heart_rate = hr_processor.get_current_bpm(
                history_measurements, time_window=1
            )
        if heart_rate is not None:
            print(f"Heart rate: {heart_rate} BPM")
        gui.update(
            current_frame_bytes,
            cv.imencode(".png", face)[1].tobytes(),
            history_measurements,
            heart_rate,
        )

    camera.release()
    gui.close()


if __name__ == "__main__":
    run()
