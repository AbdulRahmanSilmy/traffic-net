"""
This module contains the common utility functions for scheduling and running processes
that are part of a pipeline.
"""
from typing import Optional, List, Callable
import time
import schedule


def schedule_and_run_processes(
        download_interval_hours: int, 
        sleep_seconds: int, 
        list_process: List[Callable]) -> None:
    """
    Schedule and run processes sequentially in a pipeline.

    Parameters
    ----------
    download_interval_hours : int
        The interval in hours to run the processes.

    sleep_seconds : int
        The number of seconds to sleep between iterations.

    list_process : List[Callable]
        A list of functions each representing a process to be run 
        sequentially in a pipeline.
    """
    def run_processes():
        for process in list_process:
            process()
    
    run_processes()
    schedule.every(download_interval_hours).hours.do(run_processes)
    while True:
        schedule.run_pending()
        time.sleep(sleep_seconds)