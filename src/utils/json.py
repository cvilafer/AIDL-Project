import json
from typing import Any, Dict

from utils.filesystem import file_exists

def load_json(file_path: str) -> Dict[str, Any]:
    """
    Load a JSON file and return its content.

    Args:
        file_path (str): The path to the JSON file.

    Returns:
        Dict[str, Any]: The content of the JSON file.
    """
    if not file_exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    if not file_path.endswith('.json'):
        raise ValueError(f"File is not a JSON file: {file_path}")
        
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    return data

def json_stringify(data: Dict[str, Any], with_indentation: bool = False) -> None:
    """
    Convert JSON data to string format.

    Args:
        data (Dict[str, Any]): The JSON data to convert.
        indent (int): The number of spaces to use for indentation.
    """
    return json.dumps(data, indent= 4 if with_indentation else None)

def save_json(file_path: str, data: Dict[str, Any]) -> None:
    """
    Save JSON data to a file.

    Args:
        file_path (str): The path to the JSON file.
        data (Dict[str, Any]): The JSON data to save.
    """
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)
