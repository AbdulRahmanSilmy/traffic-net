"""
This module contains the ImagePreprocessor class that performs all the 
necessary preprocessing steps on the raw images. Any further preprocessing
steps can be added to this class.

The current preprocessing steps includes:
- Cropping the images vertically between the start and end pixel rows.
"""
import os
from datetime import datetime
import numpy as np 
import cv2 as cv
from tqdm import tqdm
from joblib import Parallel, delayed

from common_utils.file_manager import get_latest_folder


class ImagePreprocessor:
    """
    The ImagePreprocessor class contains methods to preprocess raw images.

    This class assumes the following folder structure for the raw images:
    ```
    source_dir
    ├── camera_1
    │   ├── date_1
    │   │   ├── image_1.jpg
    │   │   ├── image_2.jpg
    │   │   └── ...
    │   ├── date_2
    │   │   ├── image_1.jpg
    │   │   ├── image_2.jpg
    │   │   └── ...
    │   └── ...
    ├── camera_2
    │   ├── date_1
    │   │   ├── image_1.jpg
    │   │   ├── image_2.jpg
    │   │   └── ...
    │   ├── date_2
    │   │   ├── image_1.jpg
    │   │   ├── image_2.jpg
    │   │   └── ...
    │   └── ...
    └── ...
    ```
    Parameters
    ----------
    source_dir : str
        The path to the folder containing the raw images.

    destination_dir : str
        The path to the folder where the processed images will be saved.

    cameras : List[str]
        A list of camera folders to process.

    vertical_crop_start : int
        The starting pixel row to crop the images.

    vertical_crop_end : int
        The ending pixel row to crop the images.

    n_jobs : int
        The number of parallel jobs to run.
    """

    def __init__(self, 
                 source_dir: str, 
                 destination_dir: str, 
                 cameras: list[str], 
                 vertical_crop_start: int, 
                 vertical_crop_end: int, 
                 n_jobs: int):
        self.source_dir = source_dir
        self.destination_dir = destination_dir
        self.cameras = cameras
        self.vertical_crop_start = vertical_crop_start
        self.vertical_crop_end = vertical_crop_end
        self.n_jobs = n_jobs

    @staticmethod
    def vertical_crop_image(image: np.ndarray, vertical_crop_start: int, vertical_crop_end: int):
        """
        Crops the input image vertically between the start and end pixel rows.

        Parameters
        ----------
        image : np.ndarray
            The input image to be cropped. The shape of the image should be 
            (height, width, channels).

        vertical_crop_start : int
            The starting pixel row to crop the image.

        vertical_crop_end : int
            The ending pixel row to crop the image.

        Returns
        -------
        np.ndarray
            The cropped image.
        """
        return image[vertical_crop_start:vertical_crop_end, :, :]

    def process_image(self, source_dir: str, destination_dir: str):
        """
        Process an image in the source path and save the processed image to the destination path.

        Parameters
        ----------
        source_dir : str
            The path to the raw image.
        destination_dir : str
            The path to save the processed image.
        """
        image = cv.imread(source_dir)
        processed_image = self.vertical_crop_image(
            image,
            self.vertical_crop_start,
            self.vertical_crop_end)
        cv.imwrite(destination_dir, processed_image)

    def run_preprocessing(self):
        """
        Process images in the `source_dir` and save the processed images to the `destination_dir`.
        """

        for camera_folder in tqdm(self.cameras, desc='Percentage of cameras processed'):
            raw_camera_path = os.path.join(self.source_dir, camera_folder)
            raw_date_folders = os.listdir(raw_camera_path)
            processed_camera_path = os.path.join(
                self.destination_dir, camera_folder)
            os.makedirs(processed_camera_path, exist_ok=True)
            latest_processed_date_folder = get_latest_folder(
                processed_camera_path)

            if latest_processed_date_folder is None:
                run_date_folders = raw_date_folders
            else:
                run_date_folders = [
                    folder for folder in raw_date_folders if folder >= latest_processed_date_folder]

            for date_folder in tqdm(run_date_folders, desc=f'Processing camera: {camera_folder}'):
                raw_date_path = os.path.join(raw_camera_path, date_folder)
                raw_image_files = os.listdir(raw_date_path)

                processed_date_path = os.path.join(
                    self.destination_dir, camera_folder, date_folder)
                os.makedirs(processed_date_path, exist_ok=True)

                raw_image_paths = [os.path.join(
                    raw_date_path, file) for file in raw_image_files]
                processed_image_paths = [os.path.join(
                    processed_date_path, file) for file in raw_image_files]

                Parallel(n_jobs=self.n_jobs)(
                    delayed(self.process_image)(
                        raw_path, processed_path)
                    for raw_path, processed_path in zip(raw_image_paths, processed_image_paths)
                )

        print(f'Image processing completed at {datetime.now()}')
