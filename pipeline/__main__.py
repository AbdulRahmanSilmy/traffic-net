"""
This script schedules and runs the traffic anomaly detection pipeline. It downloads 
traffic images from the DriveBC website and preprocesses the images.

All parameter values are read from a configuration file named `data_ingestion_config.json`.
"""
# Imports from standard library
import json
import os

# Imports from project packages
from common_utils.file_manager import update_relative_paths
from common_utils.scheduler import schedule_and_run_processes
from data_ingestion.downloader import TrafficImageDownloader
from data_ingestion.preprocessor import ImagePreprocessor
from object_detection.generate_time_series import TrafficDataGenerator
from anomaly_detection.pipeline import AnomalyDetectionPipeline

# Define path-related constants
PIPELINE_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.dirname(PIPELINE_DIR)
CONFIG_PATH = os.path.join(PIPELINE_DIR, 'data_ingestion_config.json')

DATA_INGESTION_KEY = "data_ingestion"

OBJECT_DETECTION_KEY = "object_detection"
_DATA_GENERATOR_PARAMS_KEY = "traffic_data_generator"

ANOMALY_DETECTION_KEY = "anomaly_detection"

def main():
    """
    Main function to run the data ingestion pipeline.
    """
    # Load configuration
    with open(CONFIG_PATH, 'r', encoding='utf-8') as config_file:
        config = json.load(config_file)

    download_interval_hours = config['DOWNLOAD_INTERVAL_HOURS']
    sleep_seconds = config['SLEEP_SECONDS']
    data_ingestion_config = config[DATA_INGESTION_KEY]
    object_detection_config = config[OBJECT_DETECTION_KEY]
    anomaly_detection_config = config[ANOMALY_DETECTION_KEY]

    # Initializing data ingestion processes 
    downloader_params = data_ingestion_config['downloader_params']
    downloader_params = update_relative_paths(downloader_params, ROOT_DIR)
    downloader = TrafficImageDownloader(
        **downloader_params
    )

    preprocessor_params = data_ingestion_config['preprocessor_params']
    preprocessor_params = update_relative_paths(preprocessor_params, ROOT_DIR)
    preprocessor = ImagePreprocessor(
        **preprocessor_params
    )


    # Initializing object detection processes
    data_generator_params = object_detection_config[_DATA_GENERATOR_PARAMS_KEY]
    data_generator_params = update_relative_paths(
        data_generator_params, ROOT_DIR)
    data_generator = TrafficDataGenerator(
        **data_generator_params
    )


    # Initializing anomaly detection processes
    anomaly_detection_params = anomaly_detection_config['anomaly_detection_params']
    anomaly_detection_params = update_relative_paths(anomaly_detection_params, ROOT_DIR)
    anomaly_detection = AnomalyDetectionPipeline(
            **anomaly_detection_params
        )


    # Scheduling and running the processes
    list_process = [
        downloader.run_downloader, 
        preprocessor.run_preprocessing,
        data_generator.generate_tabular_csv,
        anomaly_detection.generate_plot_csv
    ]
    schedule_and_run_processes(download_interval_hours, sleep_seconds, list_process)


if __name__ == '__main__':
    main()







    
