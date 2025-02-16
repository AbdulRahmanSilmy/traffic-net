"""
This script downloads traffic images from the DriveBC website 
and saves them to a local folder.

The script uses the `requests` library to download images from the
DriveBC website.

The images are saved to a folder named "images" in the same directory
as the script. The images are saved in subfolders named by date, e.g.,
"images/2021-06-01". The images are named with a prefix "traffic_" and
a timestamp, e.g., "traffic_202106010000.jpg".

To run the script, you can simply execute the script in a terminal:
`python data_injestion.py`

To-do:
- Consider parallelizing the download process across cameras
"""

import os
from datetime import datetime, timedelta

import logging
from typing import List
import requests
from common_utils.file_manager import get_latest_folder, get_latest_image_timestamp


class TrafficImageDownloader:
    """
    A class to download traffic images from the DriveBC website.
    The time is assumed to be in the Pacific Time Zone. Images are uploaded
    every 2 minutes to the DriveBC website.

    Parameters:
    -----------
    image_dir: str
        The path to the directory where the images should be saved.

    cameras: List[str]
        A list of camera IDs for the traffic images.

    base_url: str
        The base URL for the DriveBC website.

    log_file: str
        The path to the log file for the downloader.


    The class imposes the following structure on the image directory:
    ```
    image_dir/
    ├── camera1/
    │   ├── 2021-06-01/
    │   │   ├── traffic_202106010000.jpg
    │   │   ├── traffic_202106010002.jpg
    │   │   └── ...
    │   ├── 2021-06-02/
    │   │   ├── traffic_202106020000.jpg
    │   │   ├── traffic_202106020002.jpg
    │   │   └── ...
    │   └── ...
    ├── camera2/
    │   ├── 2021-06-01/
    │   │   ├── traffic_202106010000.jpg
    │   │   ├── traffic_202106010002.jpg
    │   │   └── ...
    │   ├── 2021-06-02/
    │   │   ├── traffic_202106020000.jpg
    │   │   ├── traffic_202106020002.jpg
    │   │   └── ...
    │   └── ...
    └── ...
    ```
    """

    def __init__(
            self,
            image_dir: str,
            cameras: List[str],
            base_url: str,
            log_file: str):
        self.image_dir = image_dir
        self.cameras = cameras
        self.base_url = base_url
        self.log_file = log_file
        self._init_logging()

    def _init_logging(self):
        """
        Initialize logging for the downloader.
        """
        # Configure logging
        log_dir = os.path.dirname(self.log_file)
        os.makedirs(log_dir, exist_ok=True)
        logging.basicConfig(
            filename=self.log_file,  # Log to a file
            filemode='a',  # Append to the file
            format='%(asctime)s - %(levelname)s - %(message)s',
            level=logging.INFO  # Change to logging.DEBUG for more detailed output
        )

    def _download_image(self, timestamp: str, folder_path: str, camera: str) -> None:
        """
        Download an image from the DriveBC website for a given timestamp.

        Parameters:
        -----------
        timestamp: str
            The timestamp of the image in the format "YYYYMMDDHHMM".

        folder_path: str
            The path to the folder where the image should be saved.

        camera: str
            The camera ID for the image.
        """
        image_url = f"{self.base_url}{camera}/{timestamp}.jpg"
        filename = os.path.join(folder_path, f"traffic_{timestamp}.jpg")
        try:
            response = requests.get(image_url, stream=True, timeout=10)
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
                "Request error occurred while downloading image for timestamp %s: %s", timestamp, e)
        except IOError as e:
            logging.error(
                "IO error occurred while saving image for timestamp %s: %s", timestamp, e)

    def _download_time_range(self,
                             start_time: datetime,
                             end_time: datetime,
                             folder_path: str,
                             camera: str) -> None:
        """
        Download images for a time range.

        Parameters:
        -----------
        start_time: datetime
            The start time of the time range. The datetime format is "YYYY-MM-DD HH:MM".

        end_time: datetime
            The end time of the time range. The datetime format is "YYYY-MM-DD HH:MM".

        folder_path: str
            The path to the folder where the images should be saved.

        camera: str
            The camera ID for the images.
        """
        current_time = datetime.now()
        if start_time > current_time:
            logging.error("Start time is in the future.")
            return
        end_time = min(end_time, current_time)

        start_timestamp = start_time.replace(second=0, microsecond=0)

        while start_timestamp < end_time:
            timestamp_str = start_timestamp.strftime("%Y%m%d%H%M")
            self._download_image(timestamp_str, folder_path, camera)
            start_timestamp += timedelta(minutes=2)

    def _get_latest_available_time(self, current_time: datetime) -> datetime:
        """
        Get the latest available time for downloading images. 
        Images are uploaded every 2 minutes.

        Parameters:
        -----------
        current_time: datetime
            The current time. 

        Returns:
        --------
        latest_available_time: datetime
            The latest available time for downloading images.
        """
        latest_available_time = current_time - timedelta(days=1)
        latest_available_time = latest_available_time.replace(
            second=0, microsecond=0)
        latest_available_time = latest_available_time - \
            timedelta(minutes=latest_available_time.minute % 2)
        return latest_available_time

    def _get_latest_date_folder(self, camera_path: str, latest_available_date: str) -> str:
        """
        Get the latest date folder for when images should be downloaded.

        Parameters:
        -----------
        camera_path: str
            The path to the camera folder.

        latest_available_date: str
            The latest available date for downloading images.

        Returns:
        --------
        latest_date_folder: str
            The latest date folder for the camera.
        """
        latest_date_folder = get_latest_folder(camera_path)
        if (latest_date_folder is None) or (latest_date_folder < latest_available_date):
            latest_date_folder = latest_available_date
        return latest_date_folder

    def _get_latest_timestamp(
            self,
            folder_path: str,
            current_time: datetime,
            latest_available_time: datetime) -> datetime:
        """
        Get the latest timestamp for when images should be downloaded.

        Parameters:
        -----------
        folder_path: str
            The path to the folder where the images are stored.

        current_time: datetime
            The current time.

        latest_available_time: datetime
            The latest available time for downloading images.

        Returns:
        --------
        latest_timestamp: datetime
            The latest timestamp for downloading images.
        """
        latest_timestamp = get_latest_image_timestamp(folder_path)
        if latest_timestamp is None or ((current_time - latest_timestamp) > timedelta(days=1)):
            latest_timestamp = latest_available_time
        else:
            latest_timestamp += timedelta(minutes=2)
        return latest_timestamp

    def _get_time_ranges(self,
                         latest_date_folder: str,
                         current_date: str,
                         latest_timestamp: datetime,
                         current_time: datetime,
                         folder_path1: str,
                         camera_path: str):
        """
        Get the time ranges for downloading images.

        Parameters:
        -----------
        latest_date_folder: str
            The latest date folder for the camera.

        current_date: str
            The current date.

        latest_timestamp: datetime
            The latest timestamp for downloading images.

        current_time: datetime
            The current time.

        folder_path1: str
            The path to the folder where the images are stored.

        camera_path: str
            The path to the camera folder.

        Returns:
        --------
        start_times: List[datetime]
            A list of start times for the time ranges.

        end_times: List[datetime]

        folder_paths: List[str]
            A list of folder paths for the time ranges.
        """
        start_times = [latest_timestamp]
        end_times = [current_time]
        folder_paths = [folder_path1]
        
        if latest_date_folder < current_date:
            current_start_time = current_time.replace(
                hour=0, minute=0, second=0, microsecond=0)
            start_times.append(current_start_time)
            end_times = [current_start_time, current_time]
            folder_path2 = os.path.join(camera_path, current_date)
            os.makedirs(folder_path2, exist_ok=True)
            folder_paths.append(folder_path2)

        return start_times, end_times, folder_paths

    def run_downloader(self) -> None:
        """
        Run the image downloader to download traffic images from the DriveBC website.
        """
        current_time = datetime.now()
        current_date = current_time.strftime("%Y-%m-%d")
        latest_available_time = self._get_latest_available_time(current_time)
        latest_available_date = latest_available_time.strftime("%Y-%m-%d")

        camera_paths = [os.path.join(self.image_dir, num)
                        for num in self.cameras]

        for camera_path, camera in zip(camera_paths, self.cameras):
            os.makedirs(camera_path, exist_ok=True)
            latest_date_folder = self._get_latest_date_folder(
                camera_path, latest_available_date)

            folder_path1 = os.path.join(camera_path, latest_date_folder)
            os.makedirs(folder_path1, exist_ok=True)

            latest_timestamp = self._get_latest_timestamp(
                folder_path1,
                current_time,
                latest_available_time
            )

            start_times, end_times, folder_paths = self._get_time_ranges(
                latest_date_folder,
                current_date,
                latest_timestamp,
                current_time,
                folder_path1,
                camera_path
            )

            for start_time, end_time, folder_path in zip(start_times, end_times, folder_paths):
                self._download_time_range(
                    start_time, end_time, folder_path, camera)

        print(f"Download complete at {current_time}")
