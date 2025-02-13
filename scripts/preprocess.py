"""
This script preprocesses the raw images by cropping the images vertically between 
the start and end pixel rows.

The processed images are saved in the 'dat/processed_images' folder.

This script can contain all other prepocessing required for the images.
"""
import os
import json
import sys
import cv2 as cv
from tqdm import tqdm
from joblib import Parallel, delayed


ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
SRC_DIR = os.path.join(ROOT_DIR, 'src')
sys.path.append(SRC_DIR)
from data_pipeline.file_manager import get_latest_folder

# Define path-related constants
RAW_IMAGE_PATH = os.path.join(ROOT_DIR, 'dat', 'raw_images')
PROCESSED_IMAGE_PATH = os.path.join(ROOT_DIR, 'dat', 'processed_images')

# Load configuration from config.json
CONFIG_DIR = os.path.join(ROOT_DIR, 'configs')
CONFIG_PATH = os.path.join(CONFIG_DIR, 'data_pipeline.json')
with open(CONFIG_PATH, 'r', encoding='utf-8') as config_file:
    config = json.load(config_file)

CROP_START = config.get('CROP_START')
CROP_END = config.get('CROP_END')
PREPROCESS_JOBS = config.get('PREPROCESS_JOBS')


def vertical_crop_image(image, start, end):
    """
    Crops the input image vertically between the start and end pixel rows.

    Parameters
    ----------
    image : np.ndarray
        The input image to be cropped. The shape of the image should be (height, width, channels).
    start : int
        The starting pixel row for the crop.
    end : int
        The ending pixel row for the crop.

    Returns
    -------
    np.ndarray
        The cropped image.
    """
    return image[start:end, :, :]


def process_image(source_path, destination_path):
    """
    Process an image in the source path and save the processed image to the destination path.

    Parameters
    ----------
    source_path : str
        The path to the raw image.
    destination_path : str
        The path to save the processed image.

    Returns
    -------
    None
    """
    image = cv.imread(source_path)
    processed_image = vertical_crop_image(image, CROP_START, CROP_END)
    cv.imwrite(destination_path, processed_image)


def process_camera(source_path: str, destination_path: str) -> None:
    """
    Process images in the source path and save the processed images to the destination path.

    Parameters
    ----------
    source_path : str
        The path to the folder containing the raw images.
    destination_path : str
        The path to the folder where the processed images will be saved.

    Returns
    -------
    None
    """

    # List all files in the source path
    camera_folders = os.listdir(source_path)

    for camera_folder in tqdm(camera_folders,
                              desc='Percentage of cameras processed'):
        raw_camera_path = os.path.join(source_path, camera_folder)
        raw_date_folders = os.listdir(raw_camera_path)
        processed_camera_path = os.path.join(destination_path, camera_folder)
        latest_processed_date_folder = get_latest_folder(processed_camera_path)

        run_date_folders = [
            folder for folder in raw_date_folders if folder >= latest_processed_date_folder]

        for date_folder in tqdm(run_date_folders,
                                desc=f'Processing camera: {camera_folder}'):
            raw_date_path = os.path.join(raw_camera_path, date_folder)
            raw_image_files = os.listdir(raw_date_path)

            processed_date_path = os.path.join(
                destination_path, camera_folder, date_folder)
            os.makedirs(processed_date_path, exist_ok=True)

            raw_image_paths = [os.path.join(raw_date_path, file)
                               for file in raw_image_files]
            processed_image_paths = [os.path.join(
                processed_date_path, file) for file in raw_image_files]

            Parallel(n_jobs=PREPROCESS_JOBS)(
                delayed(process_image)(raw_image_path, processed_image_path)
                for raw_image_path, processed_image_path in zip(raw_image_paths, processed_image_paths)
            )

    print('Image processing completed.')


def main():
    """
    Main function to process all the images.

    Returns
    -------
    None
    """
    process_camera(RAW_IMAGE_PATH, PROCESSED_IMAGE_PATH)


if __name__ == '__main__':
    main()
