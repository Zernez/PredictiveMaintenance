"""
config.py
====================================
Utility config functions.
"""

from pathlib import Path
import yaml

def load_config(file_path: Path, file_name: str) -> dict:
    """
    Loads a YAML config file
    :param file_path: file path to use
    :param file_name: file name ot use
    :return: config file
    """
    with open(Path.joinpath(file_path, file_name), 'r', encoding='utf8') as stream:
        settings = yaml.safe_load(stream)
    return settings
