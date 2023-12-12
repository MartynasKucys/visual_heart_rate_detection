import numpy as np
from scipy.signal import butter, lfilter, find_peaks


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

    def find_hr_frequency(self, signal: np.ndarray):
        """
        Finds the peak frequency in a given signal.

        Args:
            signal: Signal from which to find the peak frequency.

        Returns:
            float: Peak frequency in Hz within the given signal.
        """
        spectrum = np.array(np.fft.fft(signal).real)
        freqs = np.array(np.fft.fftfreq(len(signal), 1 / self.fs).real)
        # max_freq_index = np.argmax(np.abs(spectrum))
        print("-"*50)
        print("s", spectrum)
        print("f", freqs)
        avg_sum_level = 3
        peak_frequency = sum(
            freqs[:avg_sum_level] * np.abs(spectrum[:avg_sum_level])) / sum(np.abs(spectrum[:avg_sum_level]))
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
        data = np.array(history[-self.fs * time_window:])
        filtered_data = self.apply_bandpass_filter(data)
        peak_frequency = self.find_hr_frequency(filtered_data)
        heart_rate = int(round(60 * peak_frequency))
        return heart_rate
