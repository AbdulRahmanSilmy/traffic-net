"""
This script schedules and runs the data ingestion pipeline. It downloads 
traffic images from the DriveBC website and preprocesses the images.

All parameter values are read from a configuration file named `config.json`.
"""
import os
import json
from typing import Optional, List, Callable
import time
import schedule

from downloader import TrafficImageDownloader
from preprocessor import ImagePreprocessor

# Define path-related constants
DATA_INGESTION_DIR = os.path.dirname(os.path.dirname(__file__))
ROOT_DIR = os.path.dirname(DATA_INGESTION_DIR)
CONFIG_PATH = os.path.join(DATA_INGESTION_DIR, 'config.json')

def update_relative_paths(params, keys, root_dir) -> None:
    """
    Update the relative paths in the parameters dictionary with the root directory.

    Parameters
    ----------
    params : dict
        The parameters dictionary.

        A dictionary as follows:
        ```
        {
            'key1': 'relative/path/to/file1',
            'key2': 'relative/path/to/file2',
            ...
        }
        ```

    keys : List[str]
        The keys to update.

    root_dir : str
        The root directory.
    """

    for key in keys:
        params[key] = os.path.join(root_dir, params[key])


def schedule_and_run_processes(
        download_interval_hours: int, 
        sleep_seconds: int, 
        list_process: List[Callable]) -> None:
    """
    Schedule and run the data ingestion processes.

    Parameters
    ----------
    download_interval_hours : int
        The interval in hours to download new images.

    sleep_seconds : int
        The number of seconds to sleep between iterations.

    list_process : List[Callable]
        A list of functions to run as part of the data ingestion pipeline.
    """
    def run_processes():
        for process in list_process:
            process()
    
    run_processes()
    schedule.every(download_interval_hours).hours.do(run_processes)
    while True:
        schedule.run_pending()
        time.sleep(sleep_seconds)

def main():
    """
    Main function to run the data ingestion pipeline.
    """
    # Load configuration
    with open(CONFIG_PATH, 'r', encoding='utf-8') as config_file:
        config = json.load(config_file)

    download_interval_hours = config['DOWNLOAD_INTERVAL_HOURS']
    sleep_seconds = config['SLEEP_SECONDS']

    downloader_params = config['downloader_params']
    downloader_paths = ['image_dir', 'log_file']
    update_relative_paths(downloader_params, downloader_paths, ROOT_DIR)
    downloader = TrafficImageDownloader(
        **downloader_params
    )

    preprocessor_params = config['preprocessor_params']
    preprocessor_paths = ['source_dir', 'destination_dir']
    update_relative_paths(preprocessor_params, preprocessor_paths, ROOT_DIR)
    preprocessor = ImagePreprocessor(
        **preprocessor_params
    )

    list_process = [downloader.run_downloader, preprocessor.run_preprocessing]

    schedule_and_run_processes(download_interval_hours, sleep_seconds, list_process)


if __name__ == '__main__':
    main()







    
