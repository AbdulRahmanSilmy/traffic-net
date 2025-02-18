"""
The main module for the object detection pipeline. It schedules and runs 
the object detection pipeline where images are converted to time series data.

All parameter values are read from a configuration file named `config.json`.
"""
import os
import json
from common_utils.file_manager import update_relative_paths
from common_utils.scheduler import schedule_and_run_processes
from object_detection.generate_time_series import TrafficDataGenerator


_DATA_GENERATOR_PARAMS_KEY = "traffic_data_generator"
DATA_INGESTION_DIR = os.path.dirname(os.path.dirname(__file__))
ROOT_DIR = os.path.dirname(DATA_INGESTION_DIR)
CONFIG_PATH = os.path.join(DATA_INGESTION_DIR, 'config.json')


def main():
    """
    Run the object detection pipeline.
    """

    with open(CONFIG_PATH, 'r') as file:
        config = json.load(file)

    download_interval_hours = config['DOWNLOAD_INTERVAL_HOURS']
    sleep_seconds = config['SLEEP_SECONDS']

    data_generator_params = config[_DATA_GENERATOR_PARAMS_KEY]
    data_generator_params = update_relative_paths(
        data_generator_params, ROOT_DIR)
    data_generator = TrafficDataGenerator(**data_generator_params)

    list_process = [data_generator.generate_tabular_csv]
    schedule_and_run_processes(
        download_interval_hours, sleep_seconds, list_process)


if __name__ == "__main__":
    main()
