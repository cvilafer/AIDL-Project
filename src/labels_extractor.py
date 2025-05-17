import argparse

from utils.logger import Logger
import utils.filesystem as filesystem
import utils.tensor as tensor_utils
import utils.json as json_utils

parser = argparse.ArgumentParser(description="Labels Extractor - Extracts labels from tensor files in a directory.")
parser.add_argument("--tensors_dir", type=str, default="./tensors", help="Relative path to the tensors directory from the current terminal location")
parser.add_argument("--labels_path", type=str, default="./labels.json", help="Relative output path to the labels JSON file")

args = parser.parse_args()

# Relative path to the tensors directory
TENSORS_DIR = args.tensors_dir

# Name of the labels JSON file
LABELS_PATH = args.labels_path

# Initialize logger
logger = Logger("LabelsExtractor")

# Get the path to the tensors directory
tensors_dir_path = filesystem.get_absolute_path(TENSORS_DIR)

# Check if the tensors directory exists
if not filesystem.directory_exists(tensors_dir_path):
    logger.error(f"Tensors directory {tensors_dir_path} does not exist.")
    exit(1)

# Get all tensor files in the directory
tensor_files = filesystem.get_files_in_directory(tensors_dir_path, extensions=[".pt"])

# Check if there are any tensor files
if not tensor_files:
    logger.error(f"No tensor files found in {tensors_dir_path}.")
    exit(1)

# Load the tensors and labels from the tensor files
tensors = []
labels = []

for tensor_file in tensor_files:
    # Load the tensor
    tensor = tensor_utils.load_tensor(tensor_file)
    
    # Append the tensor to the list
    tensors.append(tensor)
    
    # Extract the label from the file name
    label = filesystem.get_file_name(tensor_file).split("_")[0]
    
    # Append the label to the list
    labels.append(label)

    logger.debug(f"Loaded tensor from {tensor_file} with label {label}")

# Convert each label into an integer identifier depending on the label (same label = same identifier)
unique_labels = list(set(labels))
label_to_id = {label: i for i, label in enumerate(unique_labels)}
logger.info(f"Unique labels: {unique_labels}")

# Save labels with its corresponding identifiers into a JSON file
labels_path = filesystem.get_absolute_path(LABELS_PATH)
json_utils.save_json(labels_path, label_to_id)
logger.info(f"Saved labels with identifiers to {labels_path}")