from dataclasses import dataclass
from typing import List, Dict

@dataclass
class MediapipeHolisticLandmarks:
    """
    A class to represent the landmarks extracted from a video using Mediapipe Holistic.

    Attributes:
        face (List[float]): The landmarks of the face.
        pose (List[float]): The landmarks of the pose.
        left_hand (List[float]): The landmarks of the left hand.
        right_hand (List[float]): The landmarks of the right hand.
    """
    face: List[float]
    pose: List[float]
    left_hand: List[float]
    right_hand: List[float]

    @staticmethod
    def from_dict(data: Dict) -> "MediapipeHolisticLandmarks":
        """
        Create a MediapipeHolisticLandmarks instance from a dictionary.

        Args:
            data (Dict): A dictionary containing the landmarks data.

        Returns:
            MediapipeHolisticLandmarks: An instance of MediapipeHolisticLandmarks.
        """
        return MediapipeHolisticLandmarks(
            face=data.get("face", []),
            pose=data.get("pose", []),
            left_hand=data.get("left_hand", []),
            right_hand=data.get("right_hand", [])
        )
    
    def to_dict(data: List["MediapipeHolisticLandmarks"]) -> Dict:
        """
        Convert the MediapipeHolisticLandmarks instance to a dictionary.

        Args:
            data (List[MediapipeHolisticLandmarks]): A list of MediapipeHolisticLandmarks instances.
        """
        return [{
            "face": data.face,
            "pose": data.pose,
            "left_hand": data.left_hand,
            "right_hand": data.right_hand
        } for data in data]