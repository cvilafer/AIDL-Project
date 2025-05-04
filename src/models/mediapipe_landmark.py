from dataclasses import dataclass
from typing import Optional, Dict

@dataclass
class MediapipeLandmark:
    """
    A class to represent a single landmark in the Mediapipe model.

    Attributes:
        x (float): The x-coordinate of the landmark.
        y (float): The y-coordinate of the landmark.
        z (float): The z-coordinate of the landmark.
        visibility (float): The visibility of the landmark.
    """
    x: float
    y: float
    z: float
    visibility: Optional[float] = None

    @staticmethod
    def from_dict(data: Dict) -> "MediapipeLandmark":
        """
        Create a MediapipeLandmark instance from a dictionary.

        Args:
            data (Dict): A dictionary containing the landmark data.

        Returns:
            MediapipeLandmark: An instance of MediapipeLandmark.
        """
        return MediapipeLandmark(
            x=data.x,
            y=data.y,
            z=data.z,
            visibility=getattr(data, 'visibility', None)
        ) if data else None

    @staticmethod
    def to_dict(data: Dict) -> Dict:
        """
        Convert the MediapipeLandmark instance to a dictionary.
        """
        return [
            {'x': lm.x, 'y': lm.y, 'z': lm.z, 'visibility': getattr(lm, 'visibility', None)}
            for lm in data.landmark
        ] if data else []
