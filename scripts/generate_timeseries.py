"""
Script that generates a timeseries dataset for from traffic images.

The images are expected to be in following directory structure:
ROOT_PATH
    |- dat
        |- processed_images
            |- camera_number
                |- date_folder
                    |- traffic_YYYYMMDDHHMM.jpg

The script uses an object detection model to detect cars in the images and generates a tabular csv
with the following columns:
    - time: The time of the image
    - class: The class of the detected object
    - confidence: The confidence of the detection
    - num_cars: The number of cars detected
    - incoming: The number of cars incoming
    - outgoing: The number of cars outgoing

The tabular csv is saved in the following directory:
ROOT_PATH
    |- dat
        |- output
            |- traffic_<camera_number>_two_way.csv

The script can be run from the command line using the following command:
    python scripts/generate_timeseries.py 
    --camera_dir <camera_dir> 
    --columns <columns> 
    --overwrite <overwrite> 
    --best_weights_path <best_weights_path> 
    --tabular_csv_path <tabular_csv_path>
"""

import os
from typing import List
import json
from datetime import datetime
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO

# Constants
SEED = 53456
CAMERA = '147'
COLUMNS = ['time', 'class', 'confidence', 'num_cars', 'incoming', 'outgoing']
_TIME_COLUMN = 'time'
_CAMERA_COLUMN = 'camera'

# Setting up the paths
SRIPTS_PATH = os.path.dirname(os.path.abspath(__file__))
ROOT_PATH = os.path.dirname(SRIPTS_PATH)
CONFIG_PATH = os.path.join(ROOT_PATH, 'configs', 'generate_timeseries.json')
PROCESSED_IMAGE_DIR = os.path.join(ROOT_PATH, 'dat', 'processed_images')
CAMERA_DIR = os.path.join(PROCESSED_IMAGE_DIR, CAMERA)
OUTPUT_FOLDER = os.path.join(ROOT_PATH, 'dat', 'output')
TABULAR_CSV_PATH = os.path.join(OUTPUT_FOLDER, 'traffic_147_two_way.csv')
RUN_DIR = os.path.join(ROOT_PATH, 'runs', 'detect')

with open(CONFIG_PATH, 'rb') as f:
    config = json.load(f)

# Model paths
BEST_WEIGHTS_PATH = os.path.join(
    RUN_DIR, config['best_weights'], 'weights', 'best.pt')
OVERWRITE = config['overwrite']


def _create_df_row(df, result, file):
    """Creates a row for the dataframe from the output of the YOLO model"""
    # Extracting values to store in the dataframe
    cls = result.boxes.cls
    conf = result.boxes.conf
    num_cars = len(cls)
    date_string = file.split('_')[1].split('.')[0]
    date_object = datetime.strptime(date_string, '%Y%m%d%H%M')

    # Creating a row to add to the dataframe
    dict_values = [date_object, cls, conf, num_cars, 0, 0]
    dict_values = [[v] for v in dict_values]
    dict_row = dict(zip(df.columns, dict_values))
    row = pd.DataFrame(dict_row)

    # Assigning the number of cars to the incoming and outgoing columns
    vals, counts = np.unique(cls.cpu().numpy(), return_counts=True)
    for val, count in zip(vals, counts):
        row[result.names[val]] = count

    return row


def _yolo_output_to_df(df, results, files):
    """Converts the output of the YOLO model to a pandas dataframe"""
    for result, file in zip(results, files):
        row = _create_df_row(df, result, file)
        df = pd.concat([df, row], ignore_index=True)

    return df


def _camera_images_to_tabular_csv(date_folders, columns, camera_dir, model):
    """Converts the images in a camera directory to a tabular csv"""

    df = pd.DataFrame({col: [] for col in columns})

    for date_folder in tqdm(date_folders):
        date_dir = os.path.join(camera_dir, date_folder)
        image_files = os.listdir(date_dir)
        image_paths = [os.path.join(date_dir, file) for file in image_files]
        if len(image_paths) == 0:
            continue
        results = model(image_paths)
        df = _yolo_output_to_df(df, results, image_files)

    camera_num = os.path.basename(camera_dir)
    df[_CAMERA_COLUMN] = camera_num
    df[_TIME_COLUMN] = pd.to_datetime(df[_TIME_COLUMN])

    return df


def _get_old_df(tabular_csv_path, date_folders):
    """
    Gets the existing dataframe and removes the last date from it. 
    Also filters the date_folders to only include dates from the last date 
    in the dataframe onwards.
    """
    df = pd.read_csv(tabular_csv_path)
    df[_TIME_COLUMN] = pd.to_datetime(df[_TIME_COLUMN])
    min_date = df[_TIME_COLUMN].max().date()
    min_date_folder = min_date.strftime('%Y-%m-%d')
    date_folders = [
        date_folder for date_folder in date_folders if date_folder >= min_date_folder]
    remove_mask = df[_TIME_COLUMN].dt.date == min_date
    df = df[~remove_mask]

    return df, date_folders


def generate_tabular_csv(
    camera_dir: str = CAMERA_DIR,
    columns: List[str] = COLUMNS,
    overwrite: bool = OVERWRITE,
    best_weights_path: str = BEST_WEIGHTS_PATH,
    tabular_csv_path: str = TABULAR_CSV_PATH
) -> None:
    """
    Generates a tabular csv from the images in a camera directory

    Parameters
    ----------
    camera_dir : str
        The directory containing the images from the camera
    columns : List[str]
        The columns of the tabular csv
    overwrite : bool
        Whether to overwrite the existing tabular csv. If False, the 
        function will only process the images from the last date in the existing csv
    best_weights_path : str
        The path to the best weights of the model
    tabular_csv_path : str
        The path to the tabular csv
    """

    date_folders = os.listdir(camera_dir)

    if not overwrite and os.path.exists(tabular_csv_path):
        df, date_folders = _get_old_df(tabular_csv_path, date_folders)
    else:
        df = pd.DataFrame({col: [] for col in columns})

    # Load a model
    model = YOLO(best_weights_path)  # fine-tuned yolov8n model

    df_new = _camera_images_to_tabular_csv(
        date_folders, columns, camera_dir, model)
    df = pd.concat([df, df_new], ignore_index=True)

    df.to_csv(TABULAR_CSV_PATH, index=False)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Generate a timeseries dataset from traffic images.")
    parser.add_argument('--camera_dir', type=str, default=CAMERA_DIR,
                        help='The directory containing the images from the camera')
    parser.add_argument('--columns', type=str, nargs='+',
                        default=COLUMNS, help='The columns of the tabular csv')
    parser.add_argument('--overwrite', type=bool, default=OVERWRITE,
                        help='Whether to overwrite the existing tabular csv')
    parser.add_argument('--best_weights_path', type=str, default=BEST_WEIGHTS_PATH,
                        help='The path to the best weights of the model')
    parser.add_argument('--tabular_csv_path', type=str,
                        default=TABULAR_CSV_PATH, help='The path to the tabular csv')
    args = parser.parse_args()

    generate_tabular_csv(
        camera_dir=args.camera_dir,
        columns=args.columns,
        overwrite=args.overwrite,
        best_weights_path=args.best_weights_path,
        tabular_csv_path=args.tabular_csv_path
    )
