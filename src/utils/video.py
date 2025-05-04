import os
from moviepy.video.io.VideoFileClip import VideoFileClip

from utils.logger import Logger
import utils.filesystem as filesystem

class VideoEditor:
    def __init__(self, videos_dir_path: str):
        """
        Initialize the VideoEditor with the output path.

        Parameters:
            videos_dir_path (str): The path where the videos to be edited are located.

        Attributes:
            __logger (Logger): Logger instance for logging messages.
        """
        self.__logger = Logger("VideoEditor")
        self.__videos_dir_path = filesystem.get_absolute_path(videos_dir_path)

    def cut_clip(self, video_name: str, start_time: str, end_time: str):
        """
        Cut a video clip from the input video between start_time and end_time.

        Args:
            video_name (str): The name of the video to be cut.
            start_time (str): The start time for cutting the video (format: HH:MM:SS).
            end_time (str): The end time for cutting the video (format: HH:MM:SS).
        """
        if not filesystem.directory_exists(self.__videos_dir_path):
            self.__logger.warning(f"Directory {self.__videos_dir_path} does not exist. Aborting...")
            return
        
        if not filesystem.file_exists(
            filesystem.get_absolute_path(
                filesystem.join_path(self.__videos_dir_path, video_name)
            )
        ):
            self.__logger.warning(f"Video {video_name} does not exist. Aborting...")
            return

        clip = VideoFileClip(
            filesystem.get_absolute_path(
                filesystem.join_path(self.__videos_dir_path, video_name)
            )
        ).subclipped(start_time, end_time)

        clip.write_videofile(
            filesystem.get_absolute_path(
                filesystem.join_path(self.__videos_dir_path, video_name)
            ), 
            codec="libx264", 
            audio_codec="aac"
        )

        self.__logger.debug(f"Video {video_name} cut successfully")

        return