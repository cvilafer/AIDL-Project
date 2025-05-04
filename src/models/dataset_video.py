from dataclasses import dataclass
from typing import List, Dict

@dataclass
class DatasetVideo:
    """
    A class to represent a video from the MS-ASL dataset.

    Attributes:
        org_text (str): The original text of the video.
        clean_text (str): The cleaned text of the video.
        start_time (float): The start time of the video in seconds.
        signer_id (int): The ID of the signer.
        signer (int): The signer number.
        start (int): The start frame of the video.
        end (int): The end frame of the video.
        file (str): The file name of the video.
        label (int): The label of the video.
        height (float): The height of the video in pixels.
        fps (float): The frames per second of the video.
        end_time (float): The end time of the video in seconds.
        url (str): The URL of the video.
        text (str): The text of the video.
        box (List[float]): The bounding box coordinates of the signer in the video.
        width (float): The width of the video in pixels.
    """
    org_text: str
    clean_text: str
    start_time: float
    signer_id: int
    signer: int
    start: int
    end: int
    file: str
    label: int
    height: float
    fps: float
    end_time: float
    url: str
    text: str
    box: List[float]
    width: float

    @staticmethod
    def from_dict(data: Dict) -> "DatasetVideo":
        """
        Create a DatasetVideo instance from a dictionary.

        Args:
            data (Dict): A dictionary containing the video data.

        Returns:
            DatasetVideo: An instance of DatasetVideo.
        """
        return DatasetVideo(
            org_text=str(data["org_text"]),
            clean_text=str(data["clean_text"]),
            start_time=float(data["start_time"]),
            signer_id=int(data["signer_id"]),
            signer=int(data["signer"]),
            start=int(data["start"]),
            end=int(data["end"]),
            file=str(data["file"]),
            label=int(data["label"]),
            height=float(data["height"]),
            fps=float(data["fps"]),
            end_time=float(data["end_time"]),
            url=str(data["url"]),
            text=str(data["text"]),
            box=[float(num) for num in str(data["box"]).strip("[]").split(",")],
            width=float(data["width"]),
        )
    
    @staticmethod
    def to_json(data: List["DatasetVideo"]) -> List[Dict]:
        """
        Convert a list of DatasetVideo instances to a list of dictionaries.

        Args:
            data (List[DatasetVideo]): A list of DatasetVideo instances.

        Returns:
            List[Dict]: A list of dictionaries representing the video data.
        """
        return [video.__dict__ for video in data]