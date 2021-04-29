import numpy as np


class ColorHistogram:
    BIN_COUNT = 16

    def __init__(self, frames):
        self._frames = frames

    def features(self):
        features_list = np.zeros((len(self._frames), ColorHistogram.BIN_COUNT * 3), dtype=float)

        for i, frame in enumerate(self._frames):
            features_list[i] = self._color_histogram(frame)
        return features_list

    def _color_histogram(self, frame: np.array):
        _, _, channel_count = frame.shape

        # for every channel create histogram
        bin_size = 256 // ColorHistogram.BIN_COUNT
        bins = [x * bin_size for x in range(ColorHistogram.BIN_COUNT + 1)]
        histograms = [np.histogram(frame[..., c].flatten(), bins=bins)[0] for c in range(channel_count)]

        # normalize histograms
        #histograms = [hist.astype(float) / np.max(hist) for hist in histograms]

        #return np.concatenate(histograms)
        feature = np.concatenate(histograms).astype(float)
        return feature / np.max(feature)

