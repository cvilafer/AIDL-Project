import os
from typing import Optional, List

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


def create_directory(directory_path: str):
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


def rename_file(old_file_path: str, new_file_path: str):
    """
    Rename a file.

    Args:
        old_file_path (str): The current file path.
        new_file_path (str): The new file path.

    Raises:
        FileNotFoundError: If the old file path does not exist.
    """
    if file_exists(old_file_path):
        os.rename(old_file_path, new_file_path)
    else:
        raise FileNotFoundError(f"File {old_file_path} does not exist.")
    

def get_files_in_directory(directory_path: str, extensions: Optional[List[str]]) -> List[str]:
    """
    Get a list of files in a directory with specific extensions.

    Args:
        directory_path (str): The path to the directory.
        extensions (Optional[List[str]]): A list of file extensions to filter by. Defaults to None.

    Returns:
        list: A list of file paths.
    """
    if not directory_exists(directory_path):
        raise FileNotFoundError(f"Directory {directory_path} does not exist.")
    
    files = []

    for root, _, filenames in os.walk(directory_path):
        for filename in filenames:
            if extensions is None or any(filename.endswith(ext) for ext in extensions):
                files.append(os.path.join(root, filename))

    return files


def get_file_name(file_path: str) -> str:
    """
    Get the file name from a file path.

    Args:
        file_path (str): The path to the file.

    Returns:
        str: The file name.
    """
    return os.path.basename(file_path)


def split_file_name(file_path: str) -> List[str]:
    """
    Split the file name and extension from a file path.

    Args:
        file_path (str): The path to the file.

    Returns:
        list: A list containing the file name and extension.
    """
    return os.path.splitext(os.path.basename(file_path))