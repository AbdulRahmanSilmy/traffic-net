"""
This script downloads traffic images from the DriveBC website 
and saves them to a local folder.

The script uses the `requests` library to download images from the
DriveBC website. It also uses the `schedule` library to schedule the
task of downloading images every 10 minutes.

The images are saved to a folder named "images" in the same directory
as the script. The images are saved in subfolders named by date, e.g.,
"images/2021-06-01". The images are named with a prefix "traffic_" and
a timestamp, e.g., "traffic_202106010000.jpg".

To run the script, you can simply execute the script in a terminal:
`python data_injestion.py`

To-do:
- Increase period of time the run_downloader is scheduled to run
- Consider parrellizing the download process across cameras
"""

import os
import json
import argparse
from datetime import datetime, timedelta
import time
import logging
from typing import Optional
import requests
import schedule

# Define path-related constants
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
CONFIG_PATH = os.path.join(ROOT_DIR, 'config.json')
LOG_DIR = os.path.join(ROOT_DIR, 'logs')
DEFAULT_IMAGE_PATH = os.path.join(ROOT_DIR, 'dat', 'raw_images')

# Load configuration
with open(CONFIG_PATH, 'r', encoding='utf-8') as config_file:
    config = json.load(config_file)

BASE_URL = config['BASE_URL']
CAMERAS = config['CAMERAS']
# Parse command-line arguments
parser = argparse.ArgumentParser(description='Download traffic images.')
parser.add_argument('--image_path',
                    type=str,
                    default=DEFAULT_IMAGE_PATH,
                    help='Path to save images')
args = parser.parse_args()

IMAGE_PATH = args.image_path

# Configure logging
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(
        LOG_DIR, 'traffic_image_downloader.log'),  # Log to a file
    filemode='a',  # Append to the file
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO  # Change to logging.DEBUG for more detailed output
)


def download_image(timestamp: str, folder_path: str, camera: str) -> None:
    """
    Downloads an image for the given timestamp and saves it to the specified folder path.

    Parameters
    ----------
    timestamp : str
        The timestamp for which the image is to be downloaded. The format should be 'YYYYMMDDHHMM'.
    folder_path : str
        The path to the folder where the downloaded image will be saved.

    Returns
    -------
    None
    """
    image_url = f"{BASE_URL}{camera}/{timestamp}.jpg"
    filename = os.path.join(folder_path, f"traffic_{timestamp}.jpg")
    try:
        response = requests.get(image_url, stream=True,
                                timeout=10)  # Added timeout argument
        if response.status_code == 200:
            with open(filename, 'wb') as file:
                for chunk in response.iter_content(1024):
                    file.write(chunk)
            logging.info("Downloaded %s", filename)
        else:
            logging.warning(
                "Image not found for timestamp %s. Status code: %s",
                timestamp,
                response.status_code
            )
    except requests.exceptions.Timeout:
        logging.error(
            "Timeout occurred while downloading image for timestamp %s", timestamp)
    except requests.exceptions.RequestException as e:
        logging.error(
            "Request error occurred while downloading image for timestamp %s: %s",
            timestamp,
            e)
    except IOError as e:
        logging.error(
            "IO error occurred while saving image for timestamp %s: %s", timestamp, e)


def get_latest_image_timestamp(folder_path: str) -> Optional[datetime]:
    """
    Finds the latest image timestamp in the folder, or returns None if no images are found.

    Parameters
    ----------
    folder_path : str
        The path to the folder where the images are stored.

    Returns
    -------
    datetime or None
        The latest image timestamp if found, otherwise None.
    """
    timestamps = []
    for filename in os.listdir(folder_path):
        if filename.startswith("traffic_") and filename.endswith(".jpg"):
            timestamp_str = filename[8:-4]  # Extract the timestamp part
            try:
                timestamp = datetime.strptime(timestamp_str, "%Y%m%d%H%M")
                timestamps.append(timestamp)
            except ValueError:
                continue
    return max(timestamps) if timestamps else None


def get_latest_folder(camera_path: str):
    """
    Finds the latest date folder in the camera path, or returns None if no folders are found.

    Parameters
    ----------
    camera_path : str
        The path to the camera folder.

    Returns
    -------
    str or None
        The latest date folder if found, otherwise None.
    """
    date_folders = os.listdir(camera_path)
    return max(date_folders) if date_folders else None


def download_time_range(start_time: datetime, end_time: datetime, folder_path: str, camera: str) -> None:
    """
    Downloads images for a specified time range.

    Parameters
    ----------
    start_time : datetime
        The start time for the time range.
    end_time : datetime
        The end time for the time range.
    folder_path : str
        The path to the folder where the images will be saved.
    camera : str
        The camera number.

    Returns
    -------
    None
    """
    current_time = datetime.now()
    if start_time > current_time:
        logging.error("Start time is in the future.")
        return
    if end_time > current_time:
        end_time = current_time

    start_timestamp = start_time.replace(second=0, microsecond=0)
    start_timestamp = start_timestamp - \
        timedelta(minutes=start_timestamp.minute % 2)

    while start_timestamp < end_time:
        timestamp_str = start_timestamp.strftime("%Y%m%d%H%M")
        download_image(timestamp_str, folder_path, camera)
        start_timestamp += timedelta(minutes=2)


def run_downloader() -> None:
    """
    Downloads images for the latest available time range for each camera. 

    Returns
    -------
    None
    """
    current_time = datetime.now()
    current_date = current_time.strftime("%Y-%m-%d")
    # Setting the latest available time to 1 day ago
    # and round down to the nearest 2-minute interval
    # because bc drive uploads images every 2 minutes
    latest_available_time = current_time - timedelta(days=1)
    latest_available_time = latest_available_time.replace(
        second=0, microsecond=0)
    latest_available_time = latest_available_time - \
        timedelta(minutes=latest_available_time.minute % 2)
    latest_available_date = latest_available_time.strftime("%Y-%m-%d")

    camera_paths = [os.path.join(IMAGE_PATH, num) for num in CAMERAS]

    for camera_path, camera in zip(camera_paths, CAMERAS):
        os.makedirs(camera_path, exist_ok=True)

        latest_date_folder = get_latest_folder(camera_path)
        if (latest_date_folder is None) or (latest_date_folder < latest_available_date):
            latest_date_folder = latest_available_date

        folder_path1 = os.path.join(camera_path, latest_date_folder)
        os.makedirs(folder_path1, exist_ok=True)
        latest_timestamp = get_latest_image_timestamp(folder_path1)

        if latest_timestamp is None or ((current_time - latest_timestamp) > timedelta(days=1)):
            latest_timestamp = latest_available_time
        else:
            latest_timestamp += timedelta(minutes=2)

        start_times = [latest_timestamp]
        end_times = [current_time]
        folder_paths = [folder_path1]

        if latest_available_date < current_date:
            current_start_time = current_time.replace(
                hour=0, minute=0, second=0, microsecond=0)
            start_times.append(current_start_time)
            end_times = [current_start_time, current_time]
            folder_path2 = os.path.join(camera_path, current_date)
            os.makedirs(folder_path2, exist_ok=True)
            folder_paths.append(folder_path2)

        for start_time, end_time, folder_path in zip(start_times, end_times, folder_paths):
            download_time_range(start_time, end_time, folder_path, camera)


def start_downloads() -> None:
    """ 
    Start the image downloader and schedule it to run every 10 minutes. 

    Returns
    -------
    None
    """
    run_downloader()
    # Schedule the task to run every 10 minutes
    schedule.every(10).minutes.do(run_downloader)
    while True:
        schedule.run_pending()
        time.sleep(30)


if __name__ == "__main__":
    start_downloads()
