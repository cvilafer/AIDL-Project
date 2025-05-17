# Setup environment

1. Create a Conda environment which uses Python 3.9

   ```bash
   conda create -n ENVIRONMENT_NAME python=3.9 -y
   ```

2. Activate the newly created environment.

   ```bash
   conda activate ENVIRONMENT_NAME
   ```

3. Install dependencies.

   ```bash
   pip install -r requirements.txt
   ```

# Update requirements

**DO THIS ONLY** every time a new library or dependency is added to the Conda environment in order to keep a track of them and ensure
it also works for future usages of this repository.

```bash
pip list --format=freeze > requirements.txt
```

# Guide

If you are starting to use this scripts, it is highly recommended that you follow the scripts in the order specified below:

## 1. Dataset Slicer

### Description

The **Dataset Slicer** script is designed to process the MS-ASL dataset by slicing a JSON dataset file into smaller subsets based on the number of classes. This is particularly useful for managing large datasets by limiting the number of classes for training or evaluation purposes. The script groups dataset entries by their **label** property, sorts them, and slices the dataset to include only the specified number of classes. The resulting sliced dataset is saved as a new JSON file.

### Arguments

The script accepts the following command-line arguments:

- **--dataset_dir**

  - **Description**: Specifies the relative path to the MS-ASL dataset directory from the current terminal location.
  - **Type**: `str`
  - **Default**: `MS-ASL-Dataset`

  Example:

  ```bash
  --dataset_dir "./MS-ASL-Dataset"
  ```

- **--dataset_file**

  - **Description**: The name of the JSON dataset file to slice.
  - **Type**: `str`
  - **Default**: `MSASL_test.json`

  Example:

  ```bash
  --dataset_file "MSASL_train.json"
  ```

- **--max_classes**

  - **Description**: The maximum number of classes to include in the sliced dataset.
  - **Type**: `int`
  - **Default**: `100`

  Example:

  ```bash
  --max_classes 50
  ```

### Usage Example

To run the script with custom arguments:

```bash
python dataset_slicer.py --dataset_dir "./MS-ASL-Dataset" --dataset_file "MSASL_train.json" --max_classes 50
```

This command will:

1. Look for the dataset file `MSASL_train.json` in the `./MS-ASL-Dataset` directory.
2. Slice the dataset to include only the first 50 classes.
3. Save the sliced dataset as a new JSON file in the same directory.

## 2. Dataset Extractor

### Description

The **Dataset Extractor** script is designed to process the MS-ASL dataset by downloading videos from the dataset JSON file and optionally editing them to extract specific clips based on the start and end times provided in the dataset. This script also generates a report of any failed downloads, making it easier to track issues during the extraction process. The downloaded videos and reports are saved in specified directories.

### Arguments

The script accepts the following command-line arguments:

- **--dataset_dir**

  - **Description**: Specifies the relative path to the MS-ASL dataset directory from the current terminal location.
  - **Type**: `str`
  - **Default**: `./MS-ASL-Dataset`

  Example:

  ```bash
  --dataset_dir "./MS-ASL-Dataset"
  ```

- **--dataset_file**

  - **Description**: The name of the JSON dataset file to process.
  - **Type**: `str`
  - **Default**: `MSASL_test_sliced.json`

  Example:

  ```bash
  --dataset_file "MSASL_train_sliced.json"
  ```

- **--videos_dir**

  - **Description**: Specifies the relative path to the output directory where the downloaded videos will be saved.
  - **Type**: `str`
  - **Default**: `./videos`

  Example:

  ```bash
  --videos_dir "./videos"
  ```

- **--reports_dir**

  - **Description**: Specifies the relative path to the output directory where the reports will be saved.
  - **Type**: `str`
  - **Default**: `./reports`

  Example:

  ```bash
  --reports_dir "./reports"
  ```

### Usage Example

To run the script with custom arguments:

```bash
python dataset_extractor.py --dataset_dir "./MS-ASL-Dataset" --dataset_file "MSASL_train_sliced.json" --videos_dir "./videos" --reports_dir "./reports"
```

This command will:

1. Look for the dataset file `MSASL_train_sliced.json` in the `./MS-ASL-Dataset` directory.
2. Download the videos specified in the dataset file and save them in the `./videos` directory.
3. Edit the videos to extract clips based on the start and end times provided in the dataset.
4. Save a report of any failed downloads in the `./reports` directory.

## 3. Dataset Processor

### Description

The **Dataset Processor** script is designed to process the MS-ASL dataset by analyzing each video using MediaPipe Holistic and extracting the landmarks for the face, pose, and hands. These landmarks are saved as JSON files for further analysis or use in machine learning models. The script ensures that all videos in the specified directory are processed and their corresponding landmarks are saved in the output directory.

### Arguments

The script accepts the following command-line arguments:

- **--videos_dir**

  - **Description**: Specifies the relative path to the directory containing the videos to be processed.
  - **Type**: `str`
  - **Default**: `./videos`

  Example:

  ```bash
  --videos_dir "./videos"
  ```

- **--landmarks_dir**

  - **Description**: Specifies the relative path to the output directory where the landmarks of each video will be saved.
  - **Type**: `str`
  - **Default**: `./landmarks`

  Example:

  ```bash
  --landmarks_dir "./landmarks"
  ```

### Usage Example

To run the script with custom arguments:

```bash
python dataset_processor.py --videos_dir "./videos" --landmarks_dir "./landmarks"
```

This command will:

1. Look for video files in the `./videos` directory.
2. Process each video using MediaPipe Holistic to extract landmarks for the face, pose, and hands.
3. Save the extracted landmarks as JSON files in the `./landmarks` directory.
4. Log the processing status for each video, including any errors or skipped files.

## 4. Dataset Cleaner

### Description

The **Dataset Cleaner** script processes the extracted landmarks JSON files for each video, ensuring that all frames have the correct number of landmarks for the face, pose, left hand, and right hand. If any landmarks are missing in a frame, the script fills them with zero coordinates. The cleaned and flattened landmarks for each video are then converted into PyTorch tensors and saved as `.pt` files for efficient loading in future machine learning workflows.

### Arguments

The script accepts the following command-line arguments:

- **--landmarks_dir**

  - **Description**: Specifies the relative path to the directory containing the extracted landmarks JSON files.
  - **Type**: `str`
  - **Default**: `./landmarks`

  Example:

  ```bash
  --landmarks_dir "./landmarks"
  ```

- **--tensors_dir**

  - **Description**: Specifies the relative path to the output directory where the cleaned tensor files will be saved.
  - **Type**: `str`
  - **Default**: `./tensors`

  Example:

  ```bash
  --tensors_dir "./tensors"
  ```

### Usage Example

To run the script with custom arguments:

```bash
python dataset_cleaner.py --landmarks_dir "./landmarks" --tensors_dir "./tensors"
```

This command will:

1. Look for landmarks JSON files in the `./landmarks` directory.
2. Process each file, filling missing landmarks with zero coordinates as needed.
3. Flatten and convert the cleaned landmarks into PyTorch tensors.
4. Save the resulting tensors as `.pt` files in the `./tensors` directory.
5. Log the processing status for each video, including any warnings about missing landmarks.
