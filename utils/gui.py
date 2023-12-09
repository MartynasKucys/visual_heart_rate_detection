import PySimpleGUI as sg


class GUI:
    """
    This class handles the graphical user interface using PySimpleGUI.
    """

    def __init__(self, fs: int, time_window: int):
        """
        Initializes the GUI instance with a specific layout for heart rate detection visualization.

        Args:
            fs: Frame rate (sampling rate) used in calculations.
            time_window: Time window for the measurements.
        """
        self.layout = [
            [sg.Text("BPM: ", key="bpm_counter")],
            [sg.Image(filename="", key="image"), sg.Image(filename="", key="face")],
            [sg.Text("", key="calculation_status", visible=False)],
            [
                sg.ProgressBar(
                    fs * time_window, orientation="h", size=(20, 20), key="progress_bar"
                )
            ],
        ]
        self.window = sg.Window("Visual Heart Rate Detection", self.layout)

        sg.theme("Black")

    def update(
        self,
        current_frame_bytes: bytes,
        face_frame_bytes: bytes,
        history_measurements: list,
        heart_rate: int | None,
    ):
        """
        Updates elements of the GUI including images, progress bar, and BPM counter.

        Args:
            current_frame_bytes: Current frame captured from the camera.
            face_frame_bytes: Processed face image.
            history_measurements: A list of historical measurement data.
            heart_rate: Heart rate.
        """

        if len(history_measurements) < self.window["progress_bar"].MaxValue:
            self.window["progress_bar"].update(len(history_measurements))
            self.window["calculation_status"].update("Collecting data...", visible=True)
        else:
            self.window["progress_bar"].update(self.window["progress_bar"].MaxValue)
            self.window["calculation_status"].update("Data collected", visible=True)
        if heart_rate is not None and 48 < heart_rate < 180 or True:
            self.window["bpm_counter"].update("BPM: " + str(heart_rate))

        self.window["image"].update(data=current_frame_bytes)
        self.window["face"].update(data=face_frame_bytes)

    def read(self, timeout: int):
        """
        Reads the current events and values from the GUI.

        Args:
            timeout: Timeout in milliseconds for the read operation.

        Returns:
            tuple: (event name, state of GUI)
        """
        return self.window.read(timeout)

    def close(self):
        """
        Closes the GUI window and cleans up resources.
        """
        self.window.close()
