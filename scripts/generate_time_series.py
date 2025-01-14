"""
Script that generates a timeseries dataset for from traffic images.

The images are expected to be in following directory structure:
ROOT_PATH
    |- dat
        |- processed_images
            |- camera_number
                |- date_folder
                    |- traffic_YYYYMMDDHHMM.jpg

The script uses an object detection model to detect cars in the images and generates a tabular csv
with the following columns:
    - time: The time of the image
    - class: The class of the detected object
    - confidence: The confidence of the detection
    - num_cars: The number of cars detected
    - incoming: The number of cars incoming
    - outgoing: The number of cars outgoing

The tabular csv is saved in the following directory:
ROOT_PATH
    |- dat
        |- output
            |- traffic_<camera_number>_two_way.csv

The script can be run from the command line using the following command:
    python scripts/generate_timeseries.py 
    --camera_dir <camera_dir> 
    --columns <columns> 
    --overwrite <overwrite> 
    --best_weights_path <best_weights_path> 
    --tabular_csv_path <tabular_csv_path>
"""
import sys
import os
import argparse
import json

# Constants
CAMERA = '147'

# Setting up the paths
SRIPTS_PATH = os.path.dirname(os.path.abspath(__file__))
ROOT_PATH = os.path.dirname(SRIPTS_PATH)
SRC_PATH = os.path.join(ROOT_PATH, 'src')
sys.path.append(SRC_PATH)
CONFIG_PATH = os.path.join(ROOT_PATH, 'configs', 'generate_timeseries.json')
CAMERA_DIR = os.path.join(ROOT_PATH, 'dat', 'processed_images', CAMERA)
TABULAR_CSV_PATH = os.path.join(ROOT_PATH, 'dat', 'output', 'traffic_147_two_way.csv')
RUN_DIR = os.path.join(ROOT_PATH, 'runs', 'detect')

from detection.generate_time_series import generate_tabular_csv, COLUMNS

# Loading config parameters
with open(CONFIG_PATH, 'rb') as f:
    config = json.load(f)

BEST_WEIGHTS_PATH = os.path.join(
    RUN_DIR, config['best_weights'], 'weights', 'best.pt')
OVERWRITE = config['overwrite']

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Generate a timeseries dataset from traffic images.")
    parser.add_argument('--camera_dir', type=str, default=CAMERA_DIR,
                        help='The directory containing the images from the camera')
    parser.add_argument('--columns', type=str, nargs='+',
                        default=COLUMNS, help='The columns of the tabular csv')
    parser.add_argument('--overwrite', type=bool, default=OVERWRITE,
                        help='Whether to overwrite the existing tabular csv')
    parser.add_argument('--best_weights_path', type=str, default=BEST_WEIGHTS_PATH,
                        help='The path to the best weights of the model')
    parser.add_argument('--tabular_csv_path', type=str,
                        default=TABULAR_CSV_PATH, help='The path to the tabular csv')
    args = parser.parse_args()

    generate_tabular_csv(
        camera_dir=args.camera_dir,
        columns=args.columns,
        overwrite=args.overwrite,
        best_weights_path=args.best_weights_path,
        tabular_csv_path=args.tabular_csv_path
    )
