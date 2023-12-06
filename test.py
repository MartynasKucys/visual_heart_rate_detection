import numpy as np
import cv2 as cv
import PySimpleGUI as sg
from datetime import datetime
from scipy.signal import butter, lfilter
from scipy.fft import fft
from scipy.signal.windows import hamming


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


def get_face_area(img, classifier):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    faces = classifier.detectMultiScale(gray, 1.1, 4)
    if len(faces) >= 1:
        x, y, w, h = faces[0]

        # only show the forehead
        return img[y : y + h, x : x + w]
        return img[y + int(np.round(h * 0.05)) : y + int(np.round(h * 0.3)), x : x + w]
    else:
        return None


def get_face_frame_bytes(frame, face_cascade, eye_cascade, encoding=".png"):
    face = get_face_area(frame, face_cascade)

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

    # blurred_face = cv.GaussianBlur(gray_face, (0, 0), 3)
    amplified_face = cv.equalizeHist(gray_face)

    return amplified_face


def apply_bandpass_filter(signal, fs, lowcut, highcut, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype="band")
    filtered_signal = lfilter(b, a, signal)
    return filtered_signal


# def find_peak_frequency(signal, fs):
#     hamming_window = hamming(len(signal))
#     windowed_signal = signal * hamming_window
#     spectrum = np.abs(fft(windowed_signal))
#     freqs = np.fft.fftfreq(len(spectrum), 1 / fs)
#     max_freq_index = np.argmax(spectrum)
#     peak_frequency = freqs[max_freq_index]
#     return peak_frequency


def find_peak_frequency(signal, fs):
    spectrum = np.fft.fft(signal)
    freqs = np.fft.fftfreq(len(signal), 1 / fs)
    max_freq_index = np.argmax(np.abs(spectrum))
    peak_frequency = freqs[max_freq_index]

    return peak_frequency


def get_current_bpm(history, fs, time_window):
    if len(history) >= fs * time_window:
        data = np.array([item for item in history[-fs * time_window :]])
        filtered_data = apply_bandpass_filter(data, fs, 0.8, 3)
        peak_frequency = find_peak_frequency(filtered_data, fs)
        heart_rate = int(round(60 * peak_frequency))
        return heart_rate
    else:
        return None


def run():
    fs = 30
    time_window = 10

    face_cascade = cv.CascadeClassifier(
        cv.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    eye_cascade = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_eye.xml")

    gui_layout = [
        [sg.Text("BPM: ", key="bpm_counter")],
        [sg.Image(filename="", key="image"), sg.Image(filename="", key="face")],
        [sg.Text("", key="calculation_status", visible=False)],
        [
            sg.ProgressBar(
                fs * time_window, orientation="h", size=(20, 20), key="progress_bar"
            )
        ],
    ]
    window = sg.Window("Visual Heart Rate Detection", gui_layout)
    sg.theme("Black")
    cap = start_capturing()

    history_measurements = []

    while True:
        event, values = window.read(timeout=1000 // fs)

        if event == "Exit" or event == sg.WIN_CLOSED:
            break

        current_frame_bytes, current_frame = get_frame(cap)
        face_frame_bytes, face_frame = get_face_frame_bytes(
            current_frame, face_cascade, eye_cascade
        )

        amplified_face = preprocess_face(face_frame)

        if amplified_face is not None:
            average_color = np.average(amplified_face)
            history_measurements.append(average_color)

            # Update the progress bar and calculation status
        if len(history_measurements) < fs * time_window:
            window["progress_bar"].update(len(history_measurements))
            window["calculation_status"].update("Collecting data...", visible=True)
        else:
            window["progress_bar"].update(fs * time_window)
            window["calculation_status"].update("Data collected", visible=True)

            heart_rate = get_current_bpm(history_measurements, fs, time_window)

            if heart_rate is not None and 48 < heart_rate < 180:
                window["bpm_counter"].update("BPM: " + str(heart_rate))

        window["image"].update(data=current_frame_bytes)
        window["face"].update(data=face_frame_bytes)

    end_capturing(cap)


if __name__ == "__main__":
    run()
