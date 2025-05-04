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
  --dataset_dir "./datasets/MS-ASL"
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
python dataset_slicer.py --dataset_dir "./datasets/MS-ASL" --dataset_file "MSASL_train.json" --max_classes 50
```

This command will:

1. Look for the dataset file `MSASL_train.json` in the `./datasets/MS-ASL` directory.
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
  --dataset_dir "./datasets/MS-ASL"
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
  --videos_dir "./output/videos"
  ```

- **--reports_dir**

  - **Description**: Specifies the relative path to the output directory where the reports will be saved.
  - **Type**: `str`
  - **Default**: `./reports`

  Example:

  ```bash
  --reports_dir "./output/reports"
  ```

### Usage Example

To run the script with custom arguments:

```bash
python dataset_extractor.py --dataset_dir "./datasets/MS-ASL" --dataset_file "MSASL_train_sliced.json" --videos_dir "./output/videos" --reports_dir "./output/reports"
```

This command will:

1. Look for the dataset file `MSASL_train_sliced.json` in the `./datasets/MS-ASL` directory.
2. Download the videos specified in the dataset file and save them in the `./output/videos` directory.
3. Edit the videos to extract clips based on the start and end times provided in the dataset.
4. Save a report of any failed downloads in the `./output/reports` directory.
