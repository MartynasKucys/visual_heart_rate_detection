# main.py

import PySimpleGUI as sg
from datetime import datetime
from time import time
import cv2 as cv
from utils.face import FaceProcessing
from utils.camera import Camera


def run():
    gui_layout = [
        [sg.Text("BPM: ", key="bpm_counter")],
        [sg.Image(filename="", key="image"), sg.Image(filename="", key="face")],
    ]
    window = sg.Window("Visual heart rate detection", gui_layout).Finalize()
    window.Maximize()
    sg.theme("Black")

    process = Camera()
    face = FaceProcessing()

    cap, fps = process.start_capturing()
    face_cascade = cv.CascadeClassifier(
        cv.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    eye_cascade = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_eye.xml")

    history_measurements = []  # [(processed_face_frame, time_stamp), ... ]

    while True:
        event, values = window.read(timeout=20)

        if event == "Exit" or event == sg.WIN_CLOSED:
            break

        current_frame_bytes, current_frame = face.get_frame(cap)
        face_frame_bytes, face_frame = face.get_face_frame_bytes(
            current_frame, face_cascade, eye_cascade
        )

        history_measurements.append(
            (face.preprocess_face_ica(face_frame, fps), datetime.now())
        )

        window["image"].update(data=current_frame_bytes)
        window["face"].update(data=face_frame_bytes)
        window["bpm_counter"].update(
            "BPM: " + str(face.get_current_bpm(history_measurements, 10))
        )

    process.end(cap)


if __name__ == "__main__":
    run()
