"""
This module generates a tabular csv traffic data from the images in a camera directory
"""
import os
from typing import List
from datetime import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO


# Constants
_TIME_COLUMN = 'time'
_CAMERA_COLUMN = 'camera'
COLUMNS = ['time', 'class', 'confidence', 'num_cars', 'incoming', 'outgoing']

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
    camera_dir: str,
    overwrite: bool,
    best_weights_path: str,
    tabular_csv_path: str,
    columns: List[str] = COLUMNS
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

    df.to_csv(tabular_csv_path, index=False)
