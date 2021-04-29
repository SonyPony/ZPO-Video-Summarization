import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from video_info import VideoInfo


class VideoSummarizer:
    CLUSTER_COUNT_RANGE = range(10, 20)

    def __init__(self, frames, features, output_video_info: VideoInfo):
        self._frames = frames
        self._features = features
        self._out_video_info = output_video_info

    def _clusterize(self):
        # https://realpython.com/k-means-clustering-python/#how-to-perform-k-means-clustering-in-python
        # normalize features
        scaler = StandardScaler()
        normalized_features = scaler.fit_transform(self._features)

        k_means_container = list()
        prev_grad_sse = 0
        prev_sse = 0

        for cluster_count in VideoSummarizer.CLUSTER_COUNT_RANGE:
            k_means = KMeans(
                init="random",
                n_clusters=cluster_count,
                max_iter=300
            )

            k_means.fit(normalized_features)

            if cluster_count != VideoSummarizer.CLUSTER_COUNT_RANGE.start:
                if cluster_count == VideoSummarizer.CLUSTER_COUNT_RANGE.start + 1:
                    prev_grad_sse = abs(prev_sse - k_means.inertia_)
                elif prev_grad_sse * 0.6 < (grad_sse := abs(prev_sse - k_means.inertia_)):
                    prev_grad_sse = grad_sse
                else:
                    current_cluster_count = cluster_count - 1
                    break

            prev_sse = k_means.inertia_
            k_means_container.append(k_means)
        else:
            current_cluster_count = VideoSummarizer.CLUSTER_COUNT_RANGE.stop - 1
        return k_means_container[current_cluster_count - VideoSummarizer.CLUSTER_COUNT_RANGE.start].labels_, current_cluster_count

    def _identify_segments(self, labels):
        seq_len = np.zeros(labels.shape, dtype=int)
        current_label = None
        seq_start = 0

        for i, label in enumerate(labels):
            if current_label != label:
                current_label = label
                seq_start = i
                seq_len[seq_start] = 1
            else:
                seq_len[seq_start] += 1

        current_len = 0
        for i, length in enumerate(seq_len):
            if length != current_len and length != 0:
                current_len = length
            else:
                seq_len[i] = current_len
        return seq_len

    def _score_segments(self, labels, segments_len, cluster_count):
        frames_per_cluster = self._out_video_info.frame_count // cluster_count
        importance_scores = np.zeros((len(labels),), dtype=float)

        prev_label = None
        prev_frame = None
        acc_mse, segment_size, segment_start = 0, 0, 0
        use_segment = False

        for i, frame in enumerate(self._frames):
            label = labels[i]
            frame = frame / 255.

            if prev_label != label or i == len(self._frames) - 1 or segment_size >= frames_per_cluster:
                if segment_size > 1:
                    importance_scores[segment_start: segment_start + segment_size] = acc_mse / (segment_size - 1)

                use_segment = segments_len[i] > frames_per_cluster * 0.8 and prev_label != label
                acc_mse = 0
                segment_start = i
                segment_size = 1

            elif prev_label == label and use_segment:
                segment_size += 1
                acc_mse = np.sum((frame - prev_frame) ** 2)

            prev_label = label
            prev_frame = frame

        max_score = max(1., np.max(importance_scores))
        return importance_scores / max_score

    def summarize(self):
        labels, cluster_count = self._clusterize()
        segments = self._identify_segments(labels)
        scores = self._score_segments(
            labels=labels,
            segments_len=segments,
            cluster_count=cluster_count
        )

        min_segment_score = np.min(np.sort(np.unique(scores))[::-1][:cluster_count])
        selected_frames = scores >= min_segment_score

        return selected_frames, scores



