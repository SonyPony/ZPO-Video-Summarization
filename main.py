import cv2
import numpy as np
from tqdm import tqdm
from json import dumps
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from preprocess import VideoPreprocess
from videosummarizer import VideoSummarizer
from feature_generator import ColorHistogram
import os


INPUT_PATH = "../data"
OUTPUT_PATH = "../out"
OUTPUT_PATH_SCORES = "../out"

for filename in tqdm(os.listdir(INPUT_PATH)):
    video_id = filename.split(".")[0]
    input_video_path = os.path.join(INPUT_PATH, filename)
    preprocessor = VideoPreprocess(input_video_path)

    print("Preprocessing...")
    preprocessor.preprocess()
    preprocessed_frames = preprocessor.frames

    print("Generating features...")
    feature_generator = ColorHistogram(frames=preprocessed_frames)
    features = feature_generator.features()

    output_video_info = preprocessor.video_info
    output_video_info.frame_count = 600

    summarizer = VideoSummarizer(
        frames=preprocessed_frames,
        features=features,
        output_video_info=output_video_info
    )

    print("Summarizing...")
    selected_frames, scores = summarizer.summarize()

    # save
    with open(os.path.join(OUTPUT_PATH_SCORES, "{}.npy".format(video_id)), "wb") as f:
        np.save(f, scores)

    output_video = cv2.VideoWriter(
        os.path.join(OUTPUT_PATH, "{}.avi".format(video_id)),
        cv2.VideoWriter_fourcc(*"DIVX"),
        output_video_info.fps,
        (output_video_info.w, output_video_info.h)
    )
    input_video = cv2.VideoCapture(input_video_path)


    for i in range(len(preprocessed_frames)):
        _, frame = input_video.read()

        if selected_frames[i]:
            output_video.write(frame)
    input_video.release()
    output_video.release()

