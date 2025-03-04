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

    def _camera_images_to_tabular_csv(self, date_folders, start_image_files):
        """
        Converts the images in a camera directory to a tabular dataframe

        Parameters:
        -----------
        date_folders: List[str]
            The folders containing the images

        start_image_files: List[str]
            The starting image filenames per date_folder that needs to be converted to 
            time series data.
            The start image file name is formatted as `traffic_<YYYYMMDDHHMM>.jpg`.
        """
        for date_folder, start_image_file in tqdm(zip(date_folders, start_image_files), total=len(date_folders)):
            date_dir = os.path.join(self.camera_dir, date_folder)
            image_files = os.listdir(date_dir)
            image_files = [file for file in image_files if file >= start_image_file]
            image_paths = [os.path.join(date_dir, file)
                           for file in image_files]
            if len(image_paths) == 0:
                continue
            results = self.model(image_paths)
            self._yolo_output_to_df(results, image_files)

        camera_num = os.path.basename(self.camera_dir)
        self.df[self._CAMERA_COLUMN] = camera_num
        self.df[self._TIME_COLUMN] = pd.to_datetime(self.df[self._TIME_COLUMN])

    def _get_default_start_image_files(self, date_folders: List[str]) -> List[str]:
        """
        Gets the default start image files for the date folders
        The default start image file name is `traffic_<date>0000.jpg`.

        Parameters:
        -----------
        date_folders: List[str]
            The folders containing the images

        Returns:
        --------
        start_image_files: List[str]
            The default start image files for the date folders
        """
        start_image_date = [date_str.replace('-', '') + '0000' for date_str in date_folders]
        start_image_files = [f"traffic_{image_date}.jpg" for image_date in start_image_date]

        return start_image_files

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

        start_image_files: List[str]
            The starting image filenames per date_folder that needs to be converted to 
            time series data.
            The start image file name is formatted as `traffic_<YYYYMMDDHHMM>.jpg`.
        """
        df = pd.read_csv(self.tabular_csv_path)
        df[self._TIME_COLUMN] = pd.to_datetime(df[self._TIME_COLUMN])
        start_datetime = df[self._TIME_COLUMN].max()
        start_date = start_datetime.date()
        start_date_folder = start_date.strftime('%Y-%m-%d')
        latest_image_datetime = start_datetime.strftime('%Y%m%d%H%M')
        latest_image_file = f"traffic_{latest_image_datetime}.jpg"

        date_folders = [
            date_folder for date_folder in date_folders if date_folder >= start_date_folder]
        
        start_image_files = self._get_default_start_image_files(date_folders)
        start_image_files[0] = latest_image_file

        df = df.iloc[:-1,:]

        return df, date_folders, start_image_files
    
    def generate_tabular_csv(self) -> None:
        """Generates a tabular csv from the images in a camera directory"""

        date_folders = os.listdir(self.camera_dir)
        start_image_files = self._get_default_start_image_files(date_folders)
        
        if not self.overwrite and os.path.exists(self.tabular_csv_path):
            self.df, date_folders, start_image_files = self._get_old_df(date_folders)
            
        self._camera_images_to_tabular_csv(date_folders, start_image_files)
        self.df.to_csv(self.tabular_csv_path, index=False)
