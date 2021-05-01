import cv2
import numpy as np
from tqdm import tqdm
import argparse
from preprocess import VideoPreprocess
from videosummarizer import VideoSummarizer
from feature_generator import ColorHistogram
import os


parser = argparse.ArgumentParser()
parser.add_argument("input_file", type=str)
parser.add_argument("--length", type=int, required=True)
# INFO only supports avi
parser.add_argument("--out", type=str, required=True)

args = parser.parse_args()

OUTPUT_PATH_SCORES = "../out/score"

preprocessor = VideoPreprocess(args.input_file)

print("Preprocessing...")
preprocessor.preprocess()
preprocessed_frames = preprocessor.frames

print("Generating features...")
feature_generator = ColorHistogram(frames=preprocessed_frames)
features = feature_generator.features()

output_video_info = preprocessor.video_info
output_video_info.frame_count = args.length

summarizer = VideoSummarizer(
    frames=preprocessed_frames,
    features=features,
    output_video_info=output_video_info
)

print("Summarizing...")
selected_frames, scores = summarizer.summarize()

# save
#with open(os.path.join(OUTPUT_PATH_SCORES, "{}.npy".format(video_id)), "wb") as f:
#    np.save(f, scores)

output_video = cv2.VideoWriter(
    os.path.join(args.out),
    cv2.VideoWriter_fourcc(*"DIVX"),
    output_video_info.fps,
    (output_video_info.w, output_video_info.h)
)
input_video = cv2.VideoCapture(args.input_file)


for i in range(len(preprocessed_frames)):
    _, frame = input_video.read()

    if selected_frames[i]:
        output_video.write(frame)
input_video.release()
output_video.release()

