import numpy as np
import cv2 as cv
import PySimpleGUI as sg
from datetime import datetime

from time import time


def start_capturing():
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        raise Exception("Cannot open camera")
    return cap


def end_capturing(cap):
    cap.release()
    cv.destroyAllWindows()


def get_frame(cap, encoding=".png"):
    ret_val, frame = cap.read()
    img_bytes = cv.imencode(encoding, frame)[1].tobytes()
    return img_bytes, frame


# def get_face_area(img, classifier):
#     gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

#     faces = classifier.detectMultiScale(gray, 1.1, 4)
#     if len(faces) >= 1:
#         x,y,w,h =  faces[0]

#         # only show the forehead
#         return  img[y:y+h, x:x+w]
#         return img[y + int(np.round(h*0.05)) :y+ int(np.round(h*0.3)) ,x:x+w]
#     else:
#         return None


def get_face_area(img, face_cascade, eye_cascade):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    if len(faces) >= 1:
        x, y, w, h = faces[0]

        roi_face = img[y : y + h, x : x + w]
        gray_roi = cv.cvtColor(roi_face, cv.COLOR_BGR2GRAY)
        eyes = eye_cascade.detectMultiScale(gray_roi)

        for ex, ey, ew, eh in eyes:
            roi_face[ey : ey + eh, ex : ex + ew] = [0, 0, 0]

        return roi_face

    else:
        return None


def get_face_frame_bytes(frame, face_cascade, eye_cascade, encoding=".png"):
    face = get_face_area(frame, face_cascade, eye_cascade)

    if face is None:
        a = cv.imencode(encoding, np.full((100, 100, 3), fill_value=100))[1]
        return a.tobytes(), a
    else:
        a = cv.imencode(encoding, face)[1]
        return a.tobytes(), a


def preprocess_face(face_img):
    if len(face_img.shape) == 3:
        gray_face = cv.cvtColor(face_img, cv.COLOR_BGR2GRAY)
    else:
        gray_face = face_img

    blurred_face = cv.GaussianBlur(gray_face, (0, 0), 3)
    amplified_face = cv.equalizeHist(blurred_face)

    return np.average(amplified_face)


def get_current_bpm(history, time_window):
    # time window in seconds

    values_in_time_window = list()
    last_time = history[-1][1]

    for i in range(len(history) - 1, 0, -1):
        if (history[i][1] - last_time).total_seconds() <= time_window:
            values_in_time_window.append(history[i][0])
        else:
            break

    return np.average(values_in_time_window)


def run():
    gui_layout = [
        [sg.Text("BPM: ", key="bpm_counter")],
        [sg.Image(filename="", key="image"), sg.Image(filename="", key="face")],
    ]
    window = sg.Window("Visual heart rate detection", gui_layout).Finalize()
    window.Maximize()
    sg.theme("Black")
    cap = start_capturing()
    face_cascade = cv.CascadeClassifier(
        cv.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    eye_cascade = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_eye.xml")

    history_measurements = []  # [(processed_face_frame, time_stamp), ... ]

    while True:
        event, values = window.read(timeout=20)

        if event == "Exit" or event == sg.WIN_CLOSED:
            break

        current_frame_bytes, current_frame = get_frame(cap)
        face_frame_bytes, face_frame = get_face_frame_bytes(
            current_frame, face_cascade, eye_cascade
        )
        history_measurements.append((preprocess_face(face_frame), datetime.now()))

        window["image"].update(data=current_frame_bytes)
        window["face"].update(data=face_frame_bytes)
        window["bpm_counter"].update(
            "BPM: " + str(get_current_bpm(history_measurements, 10))
        )

    end_capturing(cap)


if __name__ == "__main__":
    run()
