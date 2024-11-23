"""
Subsample a days worth of data to create a smaller dataset for annotation purposes.
Data is subsampled temporally by taking a random sample during a specified time interval.
"""

import os
import argparse
from datetime import datetime
import pandas as pd
import cv2

SEED = 53456

# setting up the paths
SRC_PATH = os.path.dirname(os.path.abspath(__file__))
ROOT_PATH = os.path.dirname(SRC_PATH)


def subsample_day(camera: str, day: str, sample_dur: str = '30min', seed: int = SEED):
    """
    Subsampling images in a day for annotation. The subsampled images are saved in the annotate folder.
    root/dat/processed_images/camera/day -> root/dat/annotate/camera/day

    Parameters
    ----------
    camera : str
        Camera number
    day : str
        Day of the year YYYY-MM-DD
    sample_dur : str
        Duration for subsampling. Default is 30 minutes.
    seed : int
        Random seed for reproducibility.
    """
    print(f"Subsampling processed images for camera {camera} on day {day}")

    image_folder = os.path.join(
        ROOT_PATH, 'dat', 'processed_images', camera, day)
    annotate_folder = os.path.join(ROOT_PATH, 'dat', 'annotate', camera, day)
    os.makedirs(annotate_folder, exist_ok=True)
    image_files = [file for file in os.listdir(
        image_folder) if file.endswith('.jpg')]
    image_paths = [os.path.join(image_folder, file) for file in image_files]
    time_stamps = [file.split("_")[1].split(".")[0] for file in image_files]

    time_stamps = [
        datetime.strptime(string, '%Y%m%d%H%M') for string in time_stamps
    ]

    df = pd.DataFrame({'time': time_stamps,
                       'image_paths': image_paths})
    df.set_index('time', inplace=True)
    df_sample = df.resample(sample_dur).apply(
        lambda x: None if x.empty else x.sample(1, random_state=seed)).reset_index()

    df_sample.dropna(inplace=True)
    for path in df_sample['image_paths']:
        file_name = os.path.basename(path)
        annotate_path = os.path.join(annotate_folder, file_name)
        image = cv2.imread(path)
        cv2.imwrite(annotate_path, image)

    print(f"Subsampling complete. {len(df_sample)} images saved in {annotate_folder}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Description of your program")

    parser.add_argument('-c', '--camera', type=str, help='Camera number')
    parser.add_argument('-d', '--day', type=str,
                        help='Day of the year YYYY-MM-DD')
    parser.add_argument('-s', '--sample_dur', type=str, default='30min',
                        help='Duration for subsampling. Default is 30 minutes.')
    parser.add_argument("--seed", type=int, default=SEED)

    args = parser.parse_args()

    camera = args.camera
    day = args.day
    sample_dur = args.sample_dur
    seed = args.seed

    subsample_day(camera, day, sample_dur, seed)
