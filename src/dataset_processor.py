import argparse

from models.mediapipe_holistic_landmarks import MediapipeHolisticLandmarks
import utils.filesystem as filesystem
from utils.logger import Logger
from utils.mediapipe_holistic import MediapipeHolistic
import utils.json as json_utils

parser = argparse.ArgumentParser(description="Dataset Processor for MS-ASL - Processes each video with MediaPipe Holistic and extracts the landmarks.")
parser.add_argument("--videos_dir", type=str, default="./videos", help="Relative path to the output directory where the downloaded videos will be saved from the current terminal location")
parser.add_argument("--landmarks_dir", type=str, default="./landmarks", help="Relative path to the output directory where the landmarks of each video wiil be saved from the current terminal location")

args = parser.parse_args()

# Input directory for the downloaded videos
VIDEOS_DIR = filesystem.get_absolute_path(args.videos_dir)

# Output directory for the landmarks
LANDMARKS_DIR = filesystem.get_absolute_path(args.landmarks_dir)

# Initialize logger
logger = Logger("DatasetProcessor")

# Check if videos directory exists
if not filesystem.directory_exists(VIDEOS_DIR):
    logger.error(f"Videos directory {VIDEOS_DIR} does not exist.")
    exit(1)

# Check if the landmarks directory exists, if not create it
if not filesystem.directory_exists(LANDMARKS_DIR):
    logger.info(f"Landmarks directory {LANDMARKS_DIR} does not exist. Creating...")
    filesystem.create_directory(LANDMARKS_DIR)

# Initialize MediaPipe Holistic
mediapipe_holistic = MediapipeHolistic()

# Get the list of all video files in the videos directory
video_files_paths = filesystem.get_files_in_directory(VIDEOS_DIR, extensions=[".mp4"])

# Iterate over each video file
for video_file_path in video_files_paths:
    # Process the video and extract landmarks
    video_landmarks = mediapipe_holistic.process_video(video_file_path)

    # Check if landmarks were extracted
    if not video_landmarks:
        logger.error(f"No landmarks extracted from {video_file_path}. Skipping...")
        continue
    
    # Save the landmarks to a file
    video_file_name = filesystem.get_file_name(video_file_path)
    file_name, _ = filesystem.split_file_name(video_file_name)
    output_file_path = filesystem.join_path(LANDMARKS_DIR, f"{file_name}.json")

    json_utils.save_json(output_file_path, MediapipeHolisticLandmarks.to_dict(video_landmarks))
    logger.info(f"Processed {video_file_name} and saved landmarks to {file_name}.json")

logger.info("All videos processed successfully.")