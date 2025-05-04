import argparse

from models.dataset_video import DatasetVideo
import utils.filesystem as filesystem
import utils.json as json_utils
from utils.logger import Logger
from utils.youtube import YoutubeDownloader, DownloadResultStatus
from utils.video import VideoEditor

parser = argparse.ArgumentParser(description="Dataset Extractor for MS-ASL - Downloads and edits videos of MS-ASL dataset.")
parser.add_argument("--dataset_dir", type=str, default="./MS-ASL-Dataset", help="Relative path to the MS-ASL dataset directory from the current terminal location")
parser.add_argument("--dataset_file", type=str, default="MSASL_test_sliced.json", help="Name of the JSON dataset file to slice")
parser.add_argument("--videos_dir", type=str, default="./videos", help="Relative path to the output directory where the downloaded videos will be saved from the current terminal location")
parser.add_argument("--reports_dir", type=str, default="./reports", help="Relative path to the output directory where the reports will be saved from the current terminal location")

args = parser.parse_args()

# Relative path to the dataset directory
DATASET_DIR = filesystem.get_absolute_path(args.dataset_dir)

# Name of the dataset file
DATASET_FILE = args.dataset_file

# Output directory for the downloaded videos
VIDEOS_DIR = filesystem.get_absolute_path(args.videos_dir)

# Output directory for the reports
REPORTS_DIR = filesystem.get_absolute_path(args.reports_dir)

# Initialize logger
logger = Logger("DatasetExtractor")

# Get the path to the dataset file
dataset_file_path = filesystem.join_path(
    DATASET_DIR,
    DATASET_FILE
)

# Check if the dataset file exists
if not filesystem.file_exists(dataset_file_path):
    logger.error(f"Dataset file {dataset_file_path} does not exist.")
    exit(1)

# Load the dataset JSON file
json_dataset = json_utils.load_json(dataset_file_path)
logger.info(f"Loaded {len(json_dataset)} entries from {DATASET_FILE}")

# Setup the Youtube downloader
youtube_downloader = YoutubeDownloader(VIDEOS_DIR)

# Setup the video editor
video_editor = VideoEditor(VIDEOS_DIR)

# Download and edit each video in the dataset
# Also, register all the videos failed to download
failed_downloads = []

for label in json_dataset: 
    entry_label = json_dataset[label]
    entry_label_counter = 1

    for entry_label_video in entry_label:
        dataset_video = DatasetVideo.from_dict(entry_label_video)
        downloaded_video_name = f"{dataset_video.text}_{entry_label_counter}"

        download_result_status = youtube_downloader.download_video(
            dataset_video.url, downloaded_video_name
        )

        if download_result_status == DownloadResultStatus.SUCCESS:
            video_editor.cut_clip(
                f"{downloaded_video_name}.mp4", 
                dataset_video.start_time, 
                dataset_video.end_time
            )
        elif download_result_status == DownloadResultStatus.FAILURE:
            failed_downloads.append(dataset_video)

        entry_label_counter += 1

logger.info("Finished downloading and editing videos.")

# Check if there were any failed downloads and report them
# If there are any failed downloads, save them to a JSON file
# in the reports directory
if len(failed_downloads) > 0:
    report_file_path = filesystem.join_path(
        REPORTS_DIR,
        "failed_downloads.json"
    )

    json_utils.save_json(report_file_path, DatasetVideo.to_json(failed_downloads))
    logger.warning(f"Failed to download {len(failed_downloads)} videos. See report at {report_file_path} for more details.")