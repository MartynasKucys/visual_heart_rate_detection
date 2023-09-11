import numpy as np
import cv2 as cv
import PySimpleGUI as sg


def start_capturing():
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        raise Exception("Cannot open camera")
    return cap

def end_capturing(cap):
    cap.release()
    cv.destroyAllWindows()

def get_frame_bytes(cap, encoding=".png"):
    ret_val, frame = cap.read()
    img_bytes = cv.imencode(encoding, frame)[1].tobytes()
    return img_bytes


def run():
    gui_layout = [[sg.Text("bpm=XX")],
                  [sg.Image(filename="", key="image")]]
    window = sg.Window("Visual heart rate detection", gui_layout)
    sg.theme('Black')
    cap = start_capturing()

    while True:
        event, values = window.read(timeout=20)

        if event == "Exit" or event == sg.WIN_CLOSED:
            break

        window["image"].update(data=get_frame_bytes(cap))

    end_capturing(cap)
    
    
if __name__ == "__main__":
    run()
    