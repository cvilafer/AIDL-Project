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

## 5. Labels Extractor

### Description

The **Labels Extractor** script scans a directory of PyTorch tensor files (typically generated by the Dataset Cleaner) and extracts the unique labels from the filenames. It assigns a unique integer identifier to each label and saves the mapping as a JSON file. This is useful for preparing label-to-index mappings for classification tasks.

### Arguments

The script accepts the following command-line arguments:

- **--tensors_dir**

  - **Description**: Specifies the relative path to the directory containing the tensor files (`.pt`).
  - **Type**: `str`
  - **Default**: `./tensors`

  Example:

  ```bash
  --tensors_dir "./tensors"
  ```

- **--labels_path**

  - **Description**: Specifies the relative output path for the labels JSON file.
  - **Type**: `str`
  - **Default**: `./labels.json`

  Example:

  ```bash
  --labels_path "./labels.json"
  ```

### Usage Example

To run the script with custom arguments:

```bash
python labels_extractor.py --tensors_dir "./tensors" --labels_path "./labels.json"
```

This command will:

1. Look for tensor files in the `./tensors` directory.
2. Extract the label from each tensor file's name (the part before the first underscore).
3. Assign a unique integer identifier to each unique label.
4. Save the label-to-identifier mapping as a JSON file at `./labels.json`.
5. Log the process, including the unique labels found and the output path.

## 6. MLP Model Scripts

### 6.1. convert_video2csv.py

**Description:**
This script processes a directory of video files, extracting pose, face, and hand landmarks from each frame using MediaPipe Holistic. For each valid video (filtered by a CSV list of names), it generates a CSV file containing the landmarks for every frame, along with frame index, timestamp, and the video label.

**Key Path Variables:**

- `carpeta`: Path to the directory containing the videos to process.
- `carpeta_csv`: Path to the directory where the output CSV files will be saved.
- `csv_lista`: Path to the CSV file listing valid video names.

**Note:**
These paths are hardcoded and should be modified to match your local environment before running the script.

---

### 6.2. csv_reverse_mirroring.py

**Description:**
This script takes CSV files generated by MediaPipe Holistic and creates mirrored versions. It reflects the X coordinates (x = 1 - x) and swaps the left and right hand landmark blocks. This is useful for data augmentation in sign language recognition tasks.

**Key Path Variables:**

- The script processes all CSV files in a specified directory. You should call the `reflejar_multiples_csvs(directorio)` function, passing the path to your CSV directory.

**Note:**
Update the directory path in your function call to match your environment.

---

### 6.3. process_csv.py

**Description:**
This script preprocesses the CSV files generated by the previous step. It extracts and normalizes features such as pose, hand angles, velocities, and palm normals, then saves the processed features into new CSV files. This step is essential for preparing the data for machine learning models.

**Key Path Variables:**

- `INPUT_FOLDER`: Path to the directory containing the original CSV files.
- `OUTPUT_PROCESSED_FOLDER`: Path to the directory where the processed CSV files will be saved.

**Note:**
Both paths are hardcoded and should be updated according to your dataset location.

---

### 6.4. model_inference.py

**Description:**
This script performs frame-level sign classification using precomputed features and a Keras MLP model. It loads processed CSVs for training, validation, and testing, applies label encoding and feature scaling, trains the model, and can perform inference. The script also supports saving models and generating output images.

**Key Path Variables:**

- `PROCESSED_TRAIN_FOLDER`, `PROCESSED_VAL_FOLDER`, `PROCESSED_TEST_FOLDER`: Paths to the processed CSV directories for train, validation, and test sets.
- `csv_filtro_signos`: Path to the CSV file listing allowed sign classes.
- `SAVED_MODEL_FOLDER`: Path where trained models will be saved.
- `output_img_folder`, `video_root_folder`, `original_csv_folder`: Paths for saving output images and referencing original data.

**Note:**
All these paths are hardcoded and must be adapted to your local setup before running the script.
