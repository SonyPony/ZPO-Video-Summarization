from typing import Tuple

import cv2
import numpy as np
from math import floor
from tqdm import tqdm
from video_info import VideoInfo


class VideoPreprocess:
    SCALED_HEIGHT = 90

    def __init__(self, video_file_path: str):
        self._input_path = video_file_path
        self._frames = None

    # size in HxW format
    def preprocessed_frame_size(self) -> Tuple[int, int]:
        scaled_w = floor((VideoPreprocess.SCALED_HEIGHT / self._video_h) * self._video_w)
        return (VideoPreprocess.SCALED_HEIGHT, scaled_w)

    def _preprocess_frame(self, frame: np.array, normalize=True):
        h, w, _ = frame.shape
        scaled_h, scaled_w = self.preprocessed_frame_size()
        frame = cv2.resize(frame, dsize=(scaled_w, scaled_h))

        if normalize:
            frame = frame.astype(float) / 255.
        return frame

    def preprocess(self):
        video = cv2.VideoCapture(self._input_path)
        self._video_fps = video.get((cv2.CAP_PROP_FPS))
        self._video_w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))  # float `width`
        self._video_h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        scaled_frame_size = self.preprocessed_frame_size()
        self._frames = np.zeros((total_frame_count, ) + scaled_frame_size + (3,), dtype=float)

        for i in range(total_frame_count):
            _, frame = video.read()
            frame = self._preprocess_frame(frame,normalize=False)
            self._frames[i] = frame
        video.release()

    @property
    def frames(self) -> np.array:
        return self._frames

    @property
    def video_info(self) -> VideoInfo:
        return VideoInfo(
            w=self._video_w,
            h=self._video_h,
            fps=self._video_fps,
            frame_count=len(self._frames)
        )