import cv2
import numpy as np
from math import floor
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def preprocess_frame(frame, normalize=True):
    h, w, _ = frame.shape
    scaled_h = 90
    scaled_w = floor((scaled_h / h) * w)

    frame = cv2.resize(frame, dsize=(scaled_w, scaled_h))

    if normalize:
        frame = frame.astype(float) / 255.
    return frame


def color_histogram(frame, bin_count=16):
    _, _, channel_count = frame.shape

    # for every channel create histogram
    bin_size = 256 // bin_count
    bins = [x * bin_size for x in range(bin_count+1)]
    histograms = [np.histogram(frame[..., c].flatten(), bins=bins)[0] for c in range(channel_count)]

    # normalize histograms
    histograms = [hist.astype(float) / np.max(hist) for hist in histograms]

    return np.concatenate(histograms)

video = cv2.VideoCapture("../data/sample_2.mp4")
FPS = video.get((cv2.CAP_PROP_FPS))
FRAME_COUNT = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
#FRAME_COUNT = 10000
BIN_COUNT = 16

features = np.zeros((FRAME_COUNT, BIN_COUNT * 3), dtype=float)

print("Creating features")
for i in tqdm(range(0, FRAME_COUNT)):
    _, frame = video.read()
    frame = preprocess_frame(frame, normalize=False)
    features[i] = color_histogram(frame, bin_count=BIN_COUNT)

video.release()

# https://realpython.com/k-means-clustering-python/#how-to-perform-k-means-clustering-in-python
# normalize features
scaler = StandardScaler()
normalized_features = scaler.fit_transform(features)

k_means_container = list()
prev_grad_sse = 0
prev_sse = 0

CLUSTER_COUNT = None
CLUSTER_COUNT_START = 10
CLUSTER_COUNT_END = 20

for cluster_count in range(CLUSTER_COUNT_START, CLUSTER_COUNT_END):
    print("KMeans - {}".format(cluster_count))
    k_means = KMeans(
        init="random",
        n_clusters=cluster_count,
        max_iter=300
    )

    k_means.fit(normalized_features)

    if cluster_count == CLUSTER_COUNT_START: pass
    elif cluster_count == CLUSTER_COUNT_START + 1:
        prev_grad_sse = abs(prev_sse - k_means.inertia_)
    elif prev_grad_sse * 0.6 < (grad_sse := abs(prev_sse - k_means.inertia_)):
        prev_grad_sse = grad_sse
    else:
        CLUSTER_COUNT = cluster_count - 1
        break

    prev_sse = k_means.inertia_
    k_means_container.append(k_means)
else:
    CLUSTER_COUNT = CLUSTER_COUNT_END - 1
print("Using cluster count: {}".format(CLUSTER_COUNT))

print("Identifying short sequences")
labels = k_means_container[CLUSTER_COUNT - CLUSTER_COUNT_START].labels_
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


print("Start creating summarized video.")

seen_labels = set()
TARGET_FRAME_COUNT = 600
FRAMES_PER_CLUSTER = TARGET_FRAME_COUNT // CLUSTER_COUNT
added_frames_with_label = 0

# TODO refactor
# https://theailearner.com/2018/10/15/creating-video-from-images-using-opencv-python/
output_video = cv2.VideoWriter("../out/summarized.avi", cv2.VideoWriter_fourcc(*"DIVX"), FPS, (1280, 720))
video = cv2.VideoCapture("../data/sample_2.mp4")

for i, label in enumerate(labels):
    _, frame = video.read()

    if seq_len[i] < floor(FRAMES_PER_CLUSTER * 0.8):
        continue

    if not label in seen_labels or (current_label == label and added_frames_with_label < FRAMES_PER_CLUSTER):
        current_label = label
        seen_labels.add(label)

        output_video.write(frame)
        added_frames_with_label += 1

    else:
        current_label = None
        added_frames_with_label = 0

video.release()
output_video.release()