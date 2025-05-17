import argparse

import utils.filesystem as filesystem
import utils.json as json_utils
import utils.tensor as tensor_utils
from utils.logger import Logger

parser = argparse.ArgumentParser(description="Dataset Cleaner - Cleans up and formats a video extracted landmarks JSON file.")
parser.add_argument("--landmarks_dir", type=str, default="./landmarks", help="Relative path to the landmarks directory from the current terminal location")
parser.add_argument("--tensors_dir", type=str, default="./tensors", help="Relative path to the tensors directory from the current terminal location")

args = parser.parse_args()

# Relative path to the landmarks directory
LANDMARKS_DIR = filesystem.get_absolute_path(args.landmarks_dir)

# Check if the landmarks directory exists
if not filesystem.directory_exists(LANDMARKS_DIR):
    raise FileNotFoundError(f"Landmarks directory '{LANDMARKS_DIR}' does not exist. Please check the path.")

# Relative path to the tensors directory
TENSORS_DIR = filesystem.get_absolute_path(args.tensors_dir)

# Check if the tensors directory exists
if not filesystem.directory_exists(TENSORS_DIR):
    filesystem.create_directory(TENSORS_DIR)

# Initialize logger
logger = Logger("DatasetCleaner")

# Get all JSON files in the landmarks directory
video_landmarks_files = filesystem.get_files_in_directory(
    LANDMARKS_DIR,
    extensions=[".json"]
)

# Iterate over all JSON files
for video_landmarks_file in video_landmarks_files:
    # Get the JSON landmarks for each video
    logger.info(f"Loading {video_landmarks_file}")
    video_landmarks_json = json_utils.load_json(video_landmarks_file)

    # Get the video name from the file path
    video_name = filesystem.split_file_name(
        filesystem.get_file_name(video_landmarks_file)
    )[0]

    new_video_landmarks_json = []
    frame_counter = 1

    # Iterate over all frames in the video
    for frame_landmarks in video_landmarks_json:
        # Extract only x and y coordinates from the landmarks
        new_frame_landmarks = {
            "face": [{"x": landmark["x"], "y": landmark["y"]} for landmark in frame_landmarks["face"]],
            "pose": [{"x": landmark["x"], "y": landmark["y"]} for landmark in frame_landmarks["pose"]],
            "left_hand": [{"x": landmark["x"], "y": landmark["y"]} for landmark in frame_landmarks["left_hand"]],
            "right_hand": [{"x": landmark["x"], "y": landmark["y"]} for landmark in frame_landmarks["right_hand"]]
        }

        # Check if face is empty
        if not new_frame_landmarks["face"]:
            # Fill left_hand with 468 zero coordinates
            logger.warning(f"Face landmarks are empty in frame {frame_counter} of {video_landmarks_file}. Filling with zero coordinates...")
            new_frame_landmarks["face"] = [{"x": 0, "y": 0} for _ in range(468)]

        # Check if pose is empty
        if not new_frame_landmarks["pose"]:
            # Fill left_hand with 33 zero coordinates
            logger.warning(f"Pose landmarks are empty in frame {frame_counter} of {video_landmarks_file}. Filling with zero coordinates...")
            new_frame_landmarks["pose"] = [{"x": 0, "y": 0} for _ in range(33)]

        # Check if left_hand is empty
        if not new_frame_landmarks["left_hand"]:
            # Fill left_hand with 21 zero coordinates
            logger.warning(f"Left hand landmarks are empty in frame {frame_counter} of {video_landmarks_file}. Filling with zero coordinates...")
            new_frame_landmarks["left_hand"] = [{"x": 0, "y": 0} for _ in range(21)]

        # Check if right_hand is empty
        if not new_frame_landmarks["right_hand"]:
            # Fill right_hand with 21 zero coordinates
            logger.warning(f"Right hand landmarks are empty in frame {frame_counter} of {video_landmarks_file}. Filling with zero coordinates...")
            new_frame_landmarks["right_hand"] = [{"x": 0, "y": 0} for _ in range(21)]

        # Append the new frame landmarks to the new video landmarks JSON
        flattened_frame_landmarks = tensor_utils.flatten_frame(new_frame_landmarks)
        new_video_landmarks_json.append(flattened_frame_landmarks)

        logger.info(f"Processed frame {frame_counter} in {video_landmarks_file}")
        frame_counter += 1
    
    # Flatten the video landmarks into a 2D tensor
    flattened_video_landmarks = tensor_utils.flatten_video(new_video_landmarks_json)
    logger.info(f"Flattened video landmarks shape for {video_landmarks_file}: {flattened_video_landmarks.shape}")

    # Save the new video landmarks JSON
    tensor_utils.save_tensor(
        flattened_video_landmarks, 
        filesystem.join_path(TENSORS_DIR, f"{video_name}.pt")
    )