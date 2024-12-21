## Source Code Overview

The `src` folder contains the main source code for Traffic-Net. Below is a table explaining the purpose of each file in the `src` folder:

| File Name             | Description                                                                 |
|-----------------------|-----------------------------------------------------------------------------|
| `data_ingestion.py`   | Script responsible for ingesting data, downloading images, and saving them. |
| `preprocess.py`       | Script for basic preprocessing of raw images downloaded by `data_ingestion.py`. The preprocessed images are used for data annotation or model inference. |
| `subsample.py`        | Script that subsamples images throughout the day from preprocessed images. This helps reduce the number of images to be annotated in a day. |
| `add_annotations.py`  | Script to streamline adding annotations to the dataset, including transferring files. |
| `train_ultralytics.py`| Script to train a YOLO model from Ultralytics using annotated data.          |

For more detailed information on how to use these scripts, refer to the comments and documentation within each file.