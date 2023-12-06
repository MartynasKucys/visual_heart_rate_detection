import cv2 as cv
import numpy as np
from scipy.signal import find_peaks
from sklearn.decomposition import FastICA


class FaceProcessing:
    def __init__(self):
        pass

    def get_frame(self, cap, encoding=".png"):
        self.ret_val, self.frame = cap.read()
        self.img_bytes = cv.imencode(encoding, self.frame)[1].tobytes()
        return self.img_bytes, self.frame

    def get_face_area(self, img, face_cascade, eye_cascade):
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        if len(faces) >= 1:
            x, y, w, h = faces[0]

            self.roi_face = img[y : y + h, x : x + w]
            gray_roi = cv.cvtColor(self.roi_face, cv.COLOR_BGR2GRAY)
            eyes = eye_cascade.detectMultiScale(gray_roi)

            for ex, ey, ew, eh in eyes:
                self.roi_face[ey : ey + eh, ex : ex + ew] = [0, 0, 0]

            return self.roi_face

        else:
            return None

    def get_face_frame_bytes(self, frame, face_cascade, eye_cascade, encoding=".png"):
        face = self.get_face_area(frame, face_cascade, eye_cascade)

        if face is None:
            self.a = cv.imencode(encoding, np.full((100, 100, 3), fill_value=100))[1]
            return self.a.tobytes(), self.a
        else:
            self.a = cv.imencode(encoding, face)[1]
            return self.a.tobytes(), self.a

    def preprocess(self, face_img):
        if len(face_img.shape) == 3:
            gray_face = cv.cvtColor(face_img, cv.COLOR_BGR2GRAY)
        else:
            gray_face = face_img

        blurred_face = cv.GaussianBlur(gray_face, (0, 0), 3)
        self.amplified_face = cv.equalizeHist(blurred_face)

        return np.average(self.amplified_face)

    import numpy as np

    def preprocess_face_ica(self, face_img, fps):
        # Convert face_img to a 2D array (time x channels)
        if len(face_img.shape) == 3:
            face_img = cv.cvtColor(face_img, cv.COLOR_BGR2GRAY)

        num_frames = face_img.shape[0]

        # Reshape to (time, channels)
        face_img = face_img.reshape((num_frames, -1))

        # Convert face_img to floating point
        face_img = face_img.astype(np.float64)

        # Standardize the signal
        face_img -= np.mean(face_img, axis=0)
        face_img /= np.std(face_img, axis=0)

        # Apply ICA
        ica = FastICA(n_components=3, random_state=42)
        sources = ica.fit_transform(face_img)

        print(sources.shape)

        # Check the shape of sources
        if sources.shape[1] < 2:
            # Handle the case when sources has fewer than 2 columns
            print("Error: Insufficient components in ICA")
            return 0  # You might want to return a default value or handle the error differently

        # Find the dominant frequency in each channel
        dominant_freqs = []
        for i in range(3):
            # Take FFT of the signal
            fft_result = np.fft.fft(sources[:, i])
            fft_freqs = np.fft.fftfreq(num_frames, d=1.0 / fps)

            # Find peaks in the FFT result
            peaks, _ = find_peaks(np.abs(fft_result), height=10)

            # Filter peaks within the specified frequency range (0.67 Hz to 3 Hz)
            valid_peaks = [freq for freq in fft_freqs[peaks] if 0.67 <= freq <= 3]

            # Select the peak closest to the normal heart rate (71 bpm)
            if valid_peaks:
                selected_peak = min(valid_peaks, key=lambda x: abs(x - (71 / 60)))
                dominant_freqs.append(selected_peak)
            else:
                dominant_freqs.append(0)

        # Choose the channel with the closest frequency to the normal heart rate
        selected_channel = np.argmin(np.abs(dominant_freqs - (71 / 60)))

        # Apply the unmixing matrix to get the signal corresponding to blood volume pulse
        ica_components = ica.components_
        blood_pulse_signal = (
            sources[:, selected_channel] @ ica_components[selected_channel, :]
        )

        return np.average(blood_pulse_signal)

    def get_current_bpm(self, history, time_window):
        # time window in seconds

        self.values_in_time_window = list()
        last_time = history[-1][1]

        for i in range(len(history) - 1, 0, -1):
            if (history[i][1] - last_time).total_seconds() <= time_window:
                self.values_in_time_window.append(history[i][0])
            else:
                break

        return np.average(self.values_in_time_window)
