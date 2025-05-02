import argparse

import utils.filesystem as filesystem
import utils.json as json_utils
from utils.logger import Logger


parser = argparse.ArgumentParser(description="Dataset Slicer for MS-ASL - Slices a dataset JSON file into smaller files based on the number of classes.")
parser.add_argument("--dataset_dir", type=str, default="./MS-ASL-Dataset", help="Relative path to the MS-ASL dataset directory from the current terminal location")
parser.add_argument("--dataset_file", type=str, default="MSASL_test.json", help="Name of the JSON dataset file to slice")
parser.add_argument("--max_classes", type=int, default=100, help="Maximum number of classes to slice")

args = parser.parse_args()


# Relative path to the dataset directory
DATASET_DIR = args.dataset_dir

# Name of the dataset file
DATASET_FILE = args.dataset_file

# Maximum number of classes to slice
MAX_CLASSES = args.max_classes

# Initialize logger
logger = Logger("DatasetSlicer", is_debug_mode=False)

# Get the path to the dataset file
dataset_file_path = filesystem.join_path(
    filesystem.get_absolute_path(DATASET_DIR), 
    DATASET_FILE
)

# Check if the dataset file exists
if not filesystem.file_exists(dataset_file_path):
    logger.error(f"Dataset file {dataset_file_path} does not exist.")
    exit(1)

# Load the dataset JSON file
json_dataset = json_utils.load_json(dataset_file_path)
logger.info(f"Loaded {len(json_dataset)} entries from {DATASET_FILE}")

# Iterate through the dataset and group by the property "label" of each item
# Each group will contain the items that share the same label
grouped_dataset = {}

for item in json_dataset:
    label = item["label"]

    if label not in grouped_dataset:
        grouped_dataset[label] = []

    grouped_dataset[label].append(item)
    logger.debug(f"Grouped item with label {label}: {item}")

# Sort the grouped dataset by the label number
grouped_dataset = dict(sorted(grouped_dataset.items(), key=lambda x: int(x[0])))

logger.info(f"Grouped dataset into {len(grouped_dataset)} labels")
logger.debug(f"Grouped dataset: {json_utils.json_stringify(grouped_dataset, with_indentation=True)}")

# Slice the dataset to keep only the first MAX_CLASSES classes
sliced_dataset = {}

for i, (label, items) in enumerate(grouped_dataset.items()):
    if i >= MAX_CLASSES:
        break

    sliced_dataset[label] = items
    logger.debug(f"Sliced dataset with label {label}: {items}")

logger.info(f"Sliced dataset to {len(sliced_dataset)} labels")

# Build the output file path
output_file_path = filesystem.join_path(
    filesystem.get_absolute_path(DATASET_DIR), 
    f"{DATASET_FILE.split('.')[0]}_sliced.json"
)

# Check if the output file already exists and log a warning
if filesystem.file_exists(output_file_path):
    logger.warning(f"Output file {output_file_path} already exists. Overwriting...")

# Save the sliced dataset to a new JSON file
json_utils.save_json(output_file_path, sliced_dataset)
logger.info(f"Saved sliced dataset to {output_file_path} with {len(sliced_dataset)} labels")

logger.info("Dataset slicing completed successfully.")