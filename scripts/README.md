## Scripts Code Overview

The `scripts` folder contains the source code for Traffic-Net that does not fall under the main pipelines. Below is a table explaining the purpose of each file in the `sripts` folder:

| File Name             | Description                                                                 |
|-----------------------|-----------------------------------------------------------------------------|
| `subsample.py`        | Script that subsamples images throughout the day from preprocessed images. This helps reduce the number of images to be annotated in a day. |
| `add_annotations.py`  | Script to streamline adding annotations to the dataset, including transferring files. |
| `train_ultralytics.py`| Script to train a YOLO model from Ultralytics using annotated data.          |

For more detailed information on how to use these scripts, refer to the comments and documentation within each file.