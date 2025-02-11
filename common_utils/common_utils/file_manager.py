"""
This module contains functions for managing files and folders.
"""

import os
from datetime import datetime
from typing import Optional

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


def get_latest_folder(camera_path: str) -> Optional[str]:
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