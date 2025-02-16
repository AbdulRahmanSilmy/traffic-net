"""
The main module for the object detection pipeline. It schedules and runs 
the object detection pipeline where images are converted to time series data.

All parameter values are read from a configuration file named `config.json`.
"""
import os
import json
from typing import Optional, List, Callable
import time
import schedule

from generate_time_series import TrafficDataGenerator


_PATH_SUFFIX = "_path"
_DIR_SUFFIX = "_dir"
_DATA_GENERATOR_PARAMS_KEY = "traffic_data_generator"
DATA_INGESTION_DIR = os.path.dirname(os.path.dirname(__file__))
ROOT_DIR = os.path.dirname(DATA_INGESTION_DIR)
CONFIG_PATH = os.path.join(DATA_INGESTION_DIR, 'config.json')



def update_relative_paths(params, root_dir) -> None:
    """
    Update the relative paths in the parameters dictionary with the root directory.
    Only keys ending with '_path' or '_dir' are updated.

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
    for key, path in params.items():
        if key.endswith(_PATH_SUFFIX) or key.endswith(_DIR_SUFFIX):
            params[key] = os.path.join(root_dir, path)


def schedule_and_run_processes(
        download_interval_hours: int, 
        sleep_seconds: int, 
        list_process: List[Callable]) -> None:
    """
    Schedule and run the object detection pipeline.

    Parameters
    ----------
    download_interval_hours : int
        The interval in hours to convert images to time series data.

    sleep_seconds : int
        The number of seconds to sleep between iterations.

    list_process : List[Callable]
        A list of functions to run as part of the object detection pipeline.
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
    Run the object detection pipeline.
    """

    with open(CONFIG_PATH, 'r') as file:
        config = json.load(file)

    download_interval_hours = config['DOWNLOAD_INTERVAL_HOURS']
    sleep_seconds = config['SLEEP_SECONDS']

    data_generator_parms = config[_DATA_GENERATOR_PARAMS_KEY]
    update_relative_paths(data_generator_parms, ROOT_DIR)
    data_generator = TrafficDataGenerator(**data_generator_parms)

    list_process = [data_generator.generate_tabular_csv]
    schedule_and_run_processes(download_interval_hours, sleep_seconds, list_process)

if __name__ == "__main__":
    main()