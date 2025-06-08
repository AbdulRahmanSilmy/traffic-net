"""
This module contains functions for managing files and folders.
"""

import os
from datetime import datetime
from typing import Optional

_PATH_SUFFIX = "_path"
_DIR_SUFFIX = "_dir"


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

    Returns
    -------
    params : dict
        The updated parameters dictionary.
    """
    for key, path in params.items():
        if key.endswith(_PATH_SUFFIX) or key.endswith(_DIR_SUFFIX):
            params[key] = os.path.join(root_dir, path)

    return params
