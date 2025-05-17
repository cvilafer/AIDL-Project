import torch
from typing import List, Dict

def flatten_frame(frame_landmarks: Dict) -> List[float]:
    """
    Flattens the frame landmarks into a 1D tensor.
    
    Args:
        frame_landmarks (Dict): A dictionary containing the landmarks for a single frame.

    Returns:
        torch.Tensor: A 1D tensor containing the flattened landmarks.
    """
    # Concatenate all landmarks into a single list
    all_landmarks = frame_landmarks["face"] + frame_landmarks["pose"] + frame_landmarks["left_hand"] + frame_landmarks["right_hand"]

    # Convert every point {x, y} into a single list [x, y] and flatten the list
    flattened_frame = [coord for landmark in all_landmarks for coord in (landmark["x"], landmark["y"])]
    return flattened_frame


def flatten_video(video_landmarks: List[List[float]]) -> torch.Tensor:
    """
    Flattens the video landmarks into a 2D tensor.
    
    Args:
        video_landmarks (List[List[float]]): A list of lists containing the landmarks for each frame in the video.

    Returns:
        torch.Tensor: A 2D tensor containing the flattened landmarks for the entire video.
    """
    # Convert the list of lists into a 2D tensor
    video_tensor = torch.tensor(video_landmarks, dtype=torch.float32)
    
    # Reshape the tensor to have shape (num_frames, num_landmarks * 2)
    # Multiply by 2 because each landmark has x and y coordinates
    return video_tensor.view(video_tensor.size(0), -1)

def save_tensor(tensor: torch.Tensor, file_path: str) -> None:
    """
    Saves the tensor to a file.
    
    Args:
        tensor (torch.Tensor): The tensor to save.
        file_path (str): The path to the file where the tensor will be saved.
    """
    torch.save(tensor, file_path)


def load_tensor(file_path: str) -> torch.Tensor:
    """
    Loads the tensor from a file.
    
    Args:
        file_path (str): The path to the file from which the tensor will be loaded.

    Returns:
        torch.Tensor: The loaded tensor.
    """
    return torch.load(file_path)