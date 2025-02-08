"""
CVAT is the data annotation tool used for labelling of images. The free version 
of it does not store images when exporting the annotated dataset. This results 
in having exported zip files containing only labels. 

This script aids in data annotation process in two ways:
1. Extracts the labels from the exported zip files and appends them to the 
    existing dataset. 
2. Copies the relevant images from the annotate folder to the dataset folder.

The script is designed to be run after the annotation process is complete. 
The images are to be stored in the folder structure as follows:
    - `dat/annotate/camera/date/`
The exported zip files are to be stored in the folder structure as follows:
    - `dat/annotate/process/`
Once the zip file is processed, it is moved to the folder structure as follows:
    - `dat/annotate/done/`

To do:
1. This scripts assumes the yolo dataset folder is already created. Make it
     dynamic to create the folder if it does not exist.
2. Add logging to the script.
"""
import os
import zipfile
import shutil

# Define path-related constants
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
tranfer_function = {'copy': shutil.copy, 'move': shutil.move}


def transfer_files(src_path: str, dist_path: str, transfer_function: str) -> None:
    """
    Transfers files from the source path to the destination path using the 
    specified transfer function.
    Parameters
    ----------
    src_path : str
         The path to the source folder containing the files to be transferred.
    dist_path : str
         The path to the destination folder where the files will be transferred.
    transfer_function : str
         The transfer function to be used to transfer the files. 
         It can be either "copy" or "move".
    """
    # transfer files from source to destination
    for file in os.listdir(src_path):
        src_file_path = os.path.join(src_path, file)
        dist_file_path = os.path.join(dist_path, file)
        if os.path.exists(dist_file_path):
            print(f"File {file} already exists in {dist_path}")
            continue
        tranfer_function[transfer_function](src_file_path, dist_file_path)


def transfer_files_to_dataset_folder(extacted_zip_path: str,
                                     src_image_path: str,
                                     dataset_path: str) -> None:
    """
    Transfers:
    1. Label files from the extracted zip path to the dataset path.
    2. Relevant image files from the annotation path to the dataset path.
    Parameters
    ----------
    extacted_zip_path : str
         The path to the folder where the zip file has been extracted.
    src_image_path : str
         The path to the source folder containing the image files to be copied.
    dataset_path : str
         The path to the dataset folder where the files will be moved or copied.
    """
    # moving labels from source to destination
    # To-do: Can be either "Train" or "train"
    src_label_path = os.path.join(extacted_zip_path, 'labels', 'Train')
    if not os.path.exists(src_label_path):
        src_label_path = os.path.join(extacted_zip_path, 'labels', 'train')
    dist_label_path = os.path.join(dataset_path, 'labels', 'train')
    transfer_files(src_label_path, dist_label_path, 'move')

    # copying images from annotation to dist_path
    dist_image_path = os.path.join(dataset_path, 'images', 'train')
    transfer_files(src_image_path, dist_image_path, 'copy')


def main(root_dir: str = ROOT_DIR) -> None:
    """
    Main function to transfer files from the process folder to the dataset folder.
    Moves the processed zip files to the done folder.

    The following folder structure is assumed:
    
    ```
    root_dir
    ├── dat
    │   └── annotate
    │       ├── process
    │       ├── done
    │       ├── yolo_data
    │       └── camera_num
    │           └── date
    ```
    """
    # Define path-related constants
    annotation_path = os.path.join(root_dir, 'dat', 'annotate')

    process_path = os.path.join(annotation_path, 'process')
    done_path = os.path.join(annotation_path, 'done')
    dataset_path = os.path.join(annotation_path, 'yolo_data')

    process_zip_files = os.listdir(process_path)
    process_zip_files = [
        file for file in process_zip_files if file.endswith('.zip')]

    for zip_file in process_zip_files:
        path_to_zip_file = os.path.join(process_path, zip_file)
        directory_to_extract_to = os.path.join(
            process_path, zip_file.split('.')[0])

        with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
            zip_ref.extractall(directory_to_extract_to)

        params = zip_file.split('_')
        camera, date = params[0], params[1]
        src_image_path = os.path.join(annotation_path, camera, date)

        transfer_files_to_dataset_folder(
            directory_to_extract_to,
            src_image_path,
            dataset_path)
        shutil.rmtree(directory_to_extract_to)
        shutil.move(path_to_zip_file, os.path.join(done_path, zip_file))


if __name__ == '__main__':
    main()
