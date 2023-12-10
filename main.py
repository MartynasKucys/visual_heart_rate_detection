from utils.camera import Camera
from utils.image_processing import ImageProcessor
from utils.gui import GUI
from utils.calculations import HRSignalProcessor
import numpy as np
import PySimpleGUI as sg


def run():
    """
    Runs the heart rate calculation application.
    """
    camera = Camera()
    image_processor = ImageProcessor()
    hr_processor = HRSignalProcessor(camera.frame_rate)
    gui = GUI(camera.frame_rate, time_window=5)

    history_measurements = []

    while True:
        event, values = gui.read(timeout=1000 // camera.frame_rate)

        if event == "Exit" or event == sg.WIN_CLOSED:
            break

        current_frame_bytes, current_frame = camera.get_frame()
        face_frame_bytes, face_frame = image_processor.get_face_frame_bytes(
            current_frame
        )

        amplified_face = image_processor.preprocess_face(face_frame)

        if amplified_face is not None:
            average_color = np.average(amplified_face)
            history_measurements.append(average_color)

        heart_rate = None
        if len(history_measurements) >= camera.frame_rate:
            heart_rate = hr_processor.get_current_bpm(
                history_measurements, time_window=gui.time_window
            )

        gui.update(
            current_frame_bytes, face_frame_bytes, history_measurements, heart_rate
        )

    camera.release()
    gui.close()


if __name__ == "__main__":
    run()
