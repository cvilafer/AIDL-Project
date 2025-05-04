from pytubefix import YouTube

import utils.filesystem as filesystem
from utils.logger import Logger

class DownloadResultStatus:
    """
    Enum-like class to represent the result status of a download.
    """
    SUCCESS = "success"
    FAILURE = "failure"
    ALREADY_EXISTS = "already_exists"

class YoutubeDownloader:
    def __init__(self, download_dir_path: str):
        """
        Initialize the YoutubeDownloader with the output path.

        Parameters:
            download_dir_path (str): The path where the downloaded videos will be saved.

        Attributes:
            __logger (Logger): Logger instance for logging messages.
            __download_dir_path (str): Absolute path to the download directory.
        """
        self.__logger = Logger("YoutubeDownloader")
        self.__download_dir_path = filesystem.get_absolute_path(download_dir_path)

        if not filesystem.directory_exists(self.__download_dir_path):
            self.__logger.debug(f"Output directory {self.__download_dir_path} does not exist. Creating...")
            filesystem.create_directory(self.__download_dir_path)

    def download_video(self, video_url: str, video_name: str) -> "DownloadResultStatus":
        """
        Download a video from YouTube.

        Args:
            video_url (str): The URL of the YouTube video to download.
            video_name (str): The name to save the downloaded video as.

        Returns: 
            DownloadResultStatus: The result status of the download operation.
        """
        try:
            video_path = filesystem.join_path(self.__download_dir_path, f"{video_name}.mp4")

            if not filesystem.file_exists(video_path):
                youtube = YouTube(video_url)
                youtube_stream = youtube.streams.filter(progressive=True, file_extension='mp4').get_highest_resolution()

                self.__logger.debug(f"Downloading video {video_url} for {video_name}...")
                downloaded_video = youtube_stream.download(output_path=self.__download_dir_path)
                
                filesystem.rename_file(downloaded_video, video_path)
                self.__logger.debug(f"Video downloaded to {self.__download_dir_path} as {video_name}.mp4")

                return DownloadResultStatus.SUCCESS

            else:
                self.__logger.debug(f"Video {video_name} already exists at {self.__download_dir_path}. Skipping download...")
                return DownloadResultStatus.ALREADY_EXISTS

        except Exception as e:
            self.__logger.error(f"Failed to download video {video_url} for {video_name}. Error details: {e}")
            return DownloadResultStatus.FAILURE