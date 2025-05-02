import os

def file_exists(file_path: str) -> bool:
    """
    Check if a file exists.

    Args:
        file_path (str): The path to the file.

    Returns:
        bool: True if the file exists, False otherwise.
    """
    return os.path.isfile(file_path)

def directory_exists(directory_path: str) -> bool:
    """
    Check if a directory exists.

    Args:
        directory_path (str): The path to the directory.

    Returns:
        bool: True if the directory exists, False otherwise.
    """
    return os.path.isdir(directory_path)

def create_directory(directory_path: str) -> None:
    """
    Create a directory if it does not exist.

    Args:
        directory_path (str): The path to the directory.
    """
    if not directory_exists(directory_path):
        os.makedirs(directory_path)

def join_path(*args: str) -> str:
    """
    Join directories and file names to create a full file path.

    Args:
        *args (str): The directories and file names to join.

    Returns:
        str: The full file path.
    """
    return os.path.join(*args)


def get_absolute_path(relative_path: str) -> str:
    """
    Get the absolute path of a relative path.

    Args:
        relative_path (str): The relative path.

    Returns:
        str: The absolute path.
    """
    return os.path.abspath(relative_path)