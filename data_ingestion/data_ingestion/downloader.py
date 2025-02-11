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
- Consider parrellizing the download process across cameras
"""

import os
from datetime import datetime, timedelta

import logging
from typing import Optional, List
import requests
from common_utils.file_manager import get_latest_folder, get_latest_image_timestamp


class TrafficImageDownloader:
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
                logging.warning("Image not found for timestamp %s. Status code: %s", timestamp, response.status_code)
        except requests.exceptions.Timeout:
            logging.error("Timeout occurred while downloading image for timestamp %s", timestamp)
        except requests.exceptions.RequestException as e:
            logging.error("Request error occurred while downloading image for timestamp %s: %s", timestamp, e)
        except IOError as e:
            logging.error("IO error occurred while saving image for timestamp %s: %s", timestamp, e)

    def _download_time_range(self, start_time: datetime, end_time: datetime, folder_path: str, camera: str) -> None:
        current_time = datetime.now()
        if start_time > current_time:
            logging.error("Start time is in the future.")
            return
        if end_time > current_time:
            end_time = current_time

        start_timestamp = start_time.replace(second=0, microsecond=0)

        while start_timestamp < end_time:
            timestamp_str = start_timestamp.strftime("%Y%m%d%H%M")
            self._download_image(timestamp_str, folder_path, camera)
            start_timestamp += timedelta(minutes=2)

    def run_downloader(self) -> None:
        current_time = datetime.now()
        current_date = current_time.strftime("%Y-%m-%d")
        latest_available_time = current_time - timedelta(days=1)
        latest_available_time = latest_available_time.replace(second=0, microsecond=0)
        latest_available_time = latest_available_time - timedelta(minutes=latest_available_time.minute % 2)
        latest_available_date = latest_available_time.strftime("%Y-%m-%d")

        camera_paths = [os.path.join(self.image_dir, num) for num in self.cameras]

        for camera_path, camera in zip(camera_paths, self.cameras):
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

            if latest_date_folder < current_date:
                current_start_time = current_time.replace(hour=0, minute=0, second=0, microsecond=0)
                start_times.append(current_start_time)
                end_times = [current_start_time, current_time]
                folder_path2 = os.path.join(camera_path, current_date)
                os.makedirs(folder_path2, exist_ok=True)
                folder_paths.append(folder_path2)

            for start_time, end_time, folder_path in zip(start_times, end_times, folder_paths):
                self._download_time_range(start_time, end_time, folder_path, camera)

        print(f"Download complete at {current_time}")


