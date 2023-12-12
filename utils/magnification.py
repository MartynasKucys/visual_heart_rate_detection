import cv2 as cv
from skimage.transform import resize
import numpy as np
from scipy import fftpack, signal


class EulerMagnification(object):
    """
    This class implements Eulerian Video Magnification for revealing subtle changes in videos.
    """

    def __init__(
        self, level: int, amplification: float, fps: int, backward_frames: int = 15
    ):
        """
        Args:
            level: Number of levels in pyramids.
            amplification: Amplification factor for the magnification process.
            fps: Frame rate of the video being processed.
            backward_frames: Number of frames to look backward for temporal processing.
        """
        self.frames = []
        self.pyramids = []
        self.laplacian_pyramids = [[] for i in range(level)]
        self.level = level
        self.amplification = amplification
        self.fps = fps
        self.backward_frames = backward_frames

    def gaussian_pyramid(self, frame: np.ndarray):
        """
        Constructs a Gaussian pyramid for a given frame.

        Args:
            frame: Video frame for which the Gaussian pyramid is to be constructed.

        Returns:
            list: List of np.ndarray, each being a level of the Gaussian pyramid, downsampled sequentially.
        """
        subsample = np.copy(frame)
        pyramid_list = [subsample]

        for i in range(self.level):
            subsample = cv.pyrDown(subsample)
            pyramid_list.append(subsample)

        return pyramid_list

    def build_gaussian_pyramid(self, tensor: np.ndarray):
        """
        Constructs a Gaussian pyramid for a given frame and returns the smallest level.

        Args:
            tensor: Video frame for which the Gaussian pyramid is to be constructed.

        Returns:
            np.ndarray: Smallest level of the constructed Gaussian pyramid.
        """
        frame = tensor
        pyr = self.gaussian_pyramid(frame)
        gaussian_frame = pyr[-1]
        tensor_data = gaussian_frame

        return tensor_data

    def laplacian_pyramid(self, frame: np.ndarray):
        """
        Constructs a Laplacian pyramid for a given frame.

        Args:
            frame: Video frame for which the Laplacian pyramid is to be constructed.

        Returns:
            list: List of np.ndarray, each being a level of the Laplacian pyramid, representing the
                  difference between levels in the Gaussian pyramid.
        """
        gaussian_pyramids = self.gaussian_pyramid(frame)
        laplacian_pyramids = []

        for i in range(self.level, 0, -1):
            upper = cv.pyrUp(gaussian_pyramids[i])
            sample = cv.subtract(gaussian_pyramids[i - 1], upper)
            laplacian_pyramids.append(sample)

        return laplacian_pyramids

    def bandpass_filter(
        self, tensor: list | np.ndarray, low: float, high: float, axis: int = 0
    ):
        """
        Applies a bandpass filter to a sequence of frames.

        Args:
            tensor: Sequence of frames to be filtered.
            low: Lower frequency bound of the bandpass filter.
            high: Upper frequency bound of the bandpass filter.
            axis: Axis along which the FFT is computed.

        Returns:
            np.ndarray: Bandpass-filtered sequence of frames.
        """
        frames_arr = np.asarray(tensor, dtype=np.float64)
        fft = fftpack.fft(frames_arr, axis=axis)
        frequencies = fftpack.fftfreq(frames_arr.shape[0], d=1.0 / self.fps)
        bound_low = (np.abs(frequencies - low)).argmin()
        bound_high = (np.abs(frequencies - high)).argmin()
        fft[:bound_low] = 0
        fft[bound_high:-bound_high] = 0
        fft[-bound_low:] = 0

        iff = fftpack.ifft(fft, axis=axis)

        return np.abs(iff)

    def amplify_frame(self, frame: np.ndarray):
        """
        Amplifies a frame by the set amplification factor.

        Args:
            frame: Frame to be amplified.

        Returns:
            np.ndarray: Amplified frame.
        """
        return frame * self.amplification

    def reconstruct_frame(self, amp_frame: np.ndarray, original_frame: np.ndarray):
        """
        Reconstructs the final frame by adding the amplified frame to the original frame.

        Args:
            amp_frame: Amplified frame.
            original_frame: Original frame before amplification.

        Returns:
            np.ndarray: Reconstructed frame after adding the amplified frame to the original.
        """
        final_video = np.zeros(original_frame.shape)
        img = amp_frame
        for x in range(self.level):
            img = cv.pyrUp(img)
        img = img + original_frame
        final_video = img
        return final_video

    def magnify_color(self, frame: np.ndarray, low: float, high: float):
        """
        Magnifies the color changes in a frame.

        Args:
            frame: Current frame to be processed.
            low: Lower frequency bound.
            high: Upper frequency bound.

        Returns:
            np.ndarray: Final frame with magnified color changes.
        """
        filtered = self.bandpass_filter(
            self.pyramids[-self.backward_frames :], low, high
        )
        amplified_frames = self.amplify_frame(filtered)
        final = self.reconstruct_frame(amplified_frames[-1], frame)
        return final

    def apply_gaussian_pyramid(self, frame: np.ndarray):
        """
        Processes a frame through a Gaussian pyramid, storing the smallest level of the pyramid.

        Args:
            frame: Video frame to be processed.
        """
        pyramid = self.build_gaussian_pyramid(frame)
        self.pyramids.append(pyramid)
