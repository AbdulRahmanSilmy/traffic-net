## Source Code Overview

The `src` folder contains the main source code for Traffic-Net. Below is a table explaining the purpose of each file in the `src` folder:

| File Name            | Description                                                                 |
|----------------------|-----------------------------------------------------------------------------|
| `data_injestion.py`  | Script responsible for ingesting data, downloading images, and saving them. |
| `preprocess.py`      | Script for basic preprocessing raw images downloaded from `data_injestion.py`. The preprocessed images are to be used for data annotation or model inference |
| `subsample.py`       | Script that subsamples images across the day from preprocessed images. This aids reducing the number of images to be annotated in a day  |
| `add_annotations.py` | Script for help streamline the annotations to the dataset, including transferring files.  |
| `train_ultralytics`  | Script that trains a YOLO model from ultralytics using annotated data |


For more detailed information on how to use these scripts, refer to the comments and documentation within each file.