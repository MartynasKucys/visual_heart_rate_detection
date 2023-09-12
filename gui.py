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

def get_face(img, classifier):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    faces = classifier.detectMultiScale(gray, 1.1, 4)
    if len(faces) >= 1:
        x,y,w,h =  faces[0]
        return img[y:y+h,x:x+w]
    else:
        return None 

def get_face_frame_bytes(cap, classifier, encoding=".png"):
    ret_val, frame = cap.read()
    face = get_face(frame, classifier)
    if face is None:
        return cv.imencode(encoding, np.full((100,100,3), fill_value=100 ))[1].tobytes()
    else:
        img_bytes = cv.imencode(encoding, face)[1].tobytes()
        return img_bytes



def run():
    gui_layout = [[sg.Text("bpm=XX")],
                  [sg.Image(filename="", key="image"), sg.Image(filename="", key="face")]]
    window = sg.Window("Visual heart rate detection", gui_layout)
    sg.theme('Black')
    cap = start_capturing()
    face_cascade = cv.CascadeClassifier(cv.data.haarcascades +  'haarcascade_frontalface_default.xml')

    while True:
        event, values = window.read(timeout=20)

        if event == "Exit" or event == sg.WIN_CLOSED:
            break

        window["image"].update(data=get_frame_bytes(cap))
        window["face"].update(data=get_face_frame_bytes(cap, face_cascade))

    end_capturing(cap)
    
    
if __name__ == "__main__":
    run()
    