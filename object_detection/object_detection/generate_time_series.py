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


class TrafficDataGenerator:
    """
    Generates a tabular csv traffic data from the images in a camera directory

    Parameters:
    -----------
    camera_dir: str
        The directory containing the camera images

    best_weights_path: str
        The path to the best weights file for the YOLO model

    tabular_csv_path: str
        The path to the tabular csv file to be generated

    columns: List[str]
        The columns of the tabular csv file

    overwrite: bool, default=False
        Whether to overwrite the existing tabular csv file

    Attributes:
    -----------
    df: pd.DataFrame
        The dataframe containing the tabular csv data. It is empty by default
        contains the columns specified in the columns attribute

    model: YOLO
        The YOLO model used to detect cars in the images
    """
    _TIME_COLUMN = 'time'
    _CAMERA_COLUMN = 'camera'
    COLUMNS = ['time', 'class', 'confidence',
               'num_cars', 'incoming', 'outgoing']

    def __init__(self, 
                 camera_dir: str, 
                 best_weights_path: str, 
                 tabular_csv_path: str, 
                 columns: List[str] = None, 
                 overwrite: bool = False):
        self.camera_dir = camera_dir
        self.best_weights_path = best_weights_path
        self.tabular_csv_path = tabular_csv_path
        self.columns = columns if columns else self.COLUMNS
        self.model = YOLO(best_weights_path)
        self.overwrite = overwrite
        self.df = pd.DataFrame({col: [] for col in self.columns})

    def _create_df_row(self, result, file):
        """
        Creates a row for the dataframe from the output of the YOLO model

        Parameters:
        -----------
        result: YOLOResult
            The output of the YOLO model    

        file: str
            The name of the image file

        Returns:
        --------
        row: pd.DataFrame
            The row to be added to the dataframe
        """
        cls = result.boxes.cls
        conf = result.boxes.conf
        num_cars = len(cls)
        date_string = file.split('_')[1].split('.')[0]
        date_object = datetime.strptime(date_string, '%Y%m%d%H%M')

        dict_values = [date_object, cls, conf, num_cars, 0, 0]
        dict_values = [[v] for v in dict_values]
        dict_row = dict(zip(self.df.columns, dict_values))
        row = pd.DataFrame(dict_row)

        vals, counts = np.unique(cls.cpu().numpy(), return_counts=True)
        for val, count in zip(vals, counts):
            row[result.names[val]] = count

        return row

    def _yolo_output_to_df(self, results, files):
        """
        Converts the output of the YOLO model to time series 
        data that is appended to the df attribute. 

        Parameters:
        -----------
        results: List[YOLOResult]
            The output of the YOLO model

        files: List[str]
            The names of the image files
        """
        for result, file in zip(results, files):
            row = self._create_df_row(result, file)
            self.df = pd.concat([self.df, row], ignore_index=True)

    def _camera_images_to_tabular_csv(self, date_folders):
        """
        Converts the images in a camera directory to a tabular dataframe

        Parameters:
        -----------
        date_folders: List[str]
            The folders containing the images
        """
        for date_folder in tqdm(date_folders):
            date_dir = os.path.join(self.camera_dir, date_folder)
            image_files = os.listdir(date_dir)
            image_paths = [os.path.join(date_dir, file)
                           for file in image_files]
            if len(image_paths) == 0:
                continue
            results = self.model(image_paths)
            self._yolo_output_to_df(results, image_files)

        camera_num = os.path.basename(self.camera_dir)
        self.df[self._CAMERA_COLUMN] = camera_num
        self.df[self._TIME_COLUMN] = pd.to_datetime(self.df[self._TIME_COLUMN])

    def _get_old_df(self, date_folders: List[str]) -> List[str]:
        """
        Gets the existing dataframe and removes the last date from it.

        Parameters:
        -----------
        date_folders: List[str]
            The folders containing the images

        Returns:
        --------
        df: pd.DataFrame
            The existing dataframe with the last date removed

        date_folders: List[str]
            The folders containing the images excluding the last date
        """
        df = pd.read_csv(self.tabular_csv_path)
        df[self._TIME_COLUMN] = pd.to_datetime(df[self._TIME_COLUMN])
        min_date = df[self._TIME_COLUMN].max().date()
        min_date_folder = min_date.strftime('%Y-%m-%d')
        date_folders = [
            date_folder for date_folder in date_folders if date_folder >= min_date_folder]
        remove_mask = df[self._TIME_COLUMN].dt.date == min_date
        df = df[~remove_mask]
        return df, date_folders

    def generate_tabular_csv(self) -> None:
        """Generates a tabular csv from the images in a camera directory"""
        date_folders = os.listdir(self.camera_dir)

        if not self.overwrite and os.path.exists(self.tabular_csv_path):
            self.df, date_folders = self._get_old_df(date_folders)

        self._camera_images_to_tabular_csv(date_folders)
        self.df.to_csv(self.tabular_csv_path, index=False)
