"""
Train a YOLOv8n model and log the results to MLflow.

You can run this script from the command line by executing the following command:
```
python src/train_ultralytics.py
```
Or you can run this script from the Python interpreter by executing the following commands:
```
from src.train_ultralytics import train
train()
```
"""
import json
import os
import mlflow
from ultralytics import YOLO

# Define path-related constants
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
CONFIG_PATH = os.path.join(ROOT_DIR, 'configs', 'train_yolov8n.json')

with open(CONFIG_PATH, 'r', encoding='utf-8') as config_file:
    config = json.load(config_file)

DATA_PATH = config['data_path']
DATA_PATH = os.path.join(ROOT_DIR, DATA_PATH)

SAVE_DIR = config['save_dir']
SAVE_DIR = os.path.join(ROOT_DIR, SAVE_DIR)

WEIGHTS_PATH = config['weights_path']
TRAIN_KWARGS = config['train_kwargs']
EXPERIMENT_NAME = config['experiment_name']
MLFLOW_URI = config['mlflow_uri']


def train(data_path: str = DATA_PATH, 
          weights_path: str = WEIGHTS_PATH,
          train_kwargs: dict = TRAIN_KWARGS,
          experiment_name: str = EXPERIMENT_NAME,
          mlflow_uri: str = MLFLOW_URI) -> None:
    """
    Train a YOLOv8n model and log the results to MLflow.

    Parameters
    ----------
    data_path : str
        Path to the data directory.
    weights_path : str
        Path to the weights file.
    train_kwargs : dict
        Dictionary of training arguments.
    experiment_name : str
        Name of the MLflow experiment.

    Returns
    -------
    None
    """
    mlflow.set_tracking_uri(uri=mlflow_uri)
    mlflow.set_experiment(experiment_name)

    # Start an MLflow run
    with mlflow.start_run():
        # Log parameters
        mlflow.log_param("data_path", data_path)
        mlflow.log_param("weights_path", weights_path)
        mlflow.log_params(train_kwargs)

        model = YOLO(weights_path)

        # Train the model
        model.train(data=data_path, save_dir=SAVE_DIR, **train_kwargs)

        metrics = model.val(save=True)

        mlflow.log_param("save_dir", metrics.save_dir)


if __name__ == '__main__':
    train()
