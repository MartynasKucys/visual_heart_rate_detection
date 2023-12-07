import numpy as np
from scipy.signal import butter, lfilter


class HRSignalProcessor:
    """
    This class processes heart rate signals.
    """

    def __init__(self, fs: int):
        """
        Initializes the HRSignalProcessor with a specified frame rate (sampling rate).

        Args:
            fs: Frame rate (or sampling rate) of the signal, in Hz.
        """
        self.fs = fs

    def apply_bandpass_filter(
        self,
        signal: np.ndarray,
        lowcut: float = 0.8,
        highcut: float = 3,
        order: int = 5,
    ):
        """
        Applies a Butterworth bandpass filter to a given signal.

        Args:
            signal: Input signal to be filtered.
            lowcut: Lower frequency bound of the bandpass filter in Hz.
            highcut: Upper frequency bound of the bandpass filter in Hz.
            order: Order of the filter (higher order = sharper frequency cutoff).

        Returns:
            np.ndarray: Bandpass-filtered signal.
        """
        nyquist = 0.5 * self.fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype="band")
        filtered_signal = lfilter(b, a, signal)
        return filtered_signal

    def find_peak_frequency(self, signal: np.ndarray):
        """
        Finds the peak frequency in a given signal.

        Args:
            signal: Signal from which to find the peak frequency.

        Returns:
            float: Peak frequency in Hz within the given signal.
        """
        spectrum = np.fft.fft(signal)
        freqs = np.fft.fftfreq(len(signal), 1 / self.fs)
        max_freq_index = np.argmax(np.abs(spectrum))
        peak_frequency = freqs[max_freq_index]
        return peak_frequency

    def get_current_bpm(self, history: list, time_window: int):
        """
        Calculates the current beats per minute (BPM) from a given history of signal data.

        Args:
            history: Historical signal data used to calculate the heart rate.
            time_window: Time window in seconds for the BPM calculation.

        Returns:
            int: Estimated current heart rate.
        """
        data = np.array(history[-self.fs * time_window :])
        filtered_data = self.apply_bandpass_filter(data)
        peak_frequency = self.find_peak_frequency(filtered_data)
        heart_rate = int(round(60 * peak_frequency))
        return heart_rate
