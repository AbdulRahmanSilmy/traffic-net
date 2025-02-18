"""
This script schedules and runs the data ingestion pipeline. It downloads 
traffic images from the DriveBC website and preprocesses the images.

All parameter values are read from a configuration file named `config.json`.
"""
# Imports from standard library
import json
import os

# Imports from project packages
from common_utils.file_manager import update_relative_paths
from common_utils.scheduler import schedule_and_run_processes
from data_ingestion.downloader import TrafficImageDownloader
from data_ingestion.preprocessor import ImagePreprocessor


# Define path-related constants
DATA_INGESTION_DIR = os.path.dirname(os.path.dirname(__file__))
ROOT_DIR = os.path.dirname(DATA_INGESTION_DIR)
CONFIG_PATH = os.path.join(DATA_INGESTION_DIR, 'config.json')


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
    downloader_params = update_relative_paths(downloader_params, ROOT_DIR)
    downloader = TrafficImageDownloader(
        **downloader_params
    )

    preprocessor_params = config['preprocessor_params']
    preprocessor_params = update_relative_paths(preprocessor_params, ROOT_DIR)
    preprocessor = ImagePreprocessor(
        **preprocessor_params
    )

    list_process = [downloader.run_downloader, preprocessor.run_preprocessing]
    schedule_and_run_processes(download_interval_hours, sleep_seconds, list_process)


if __name__ == '__main__':
    main()







    
