# Importing required libraries
import os                             # OS-level operations like file/directory handling
from box.exceptions import BoxValueError  # Handles Box library value errors
import yaml                           # For reading and writing YAML files
from nephroscan_ai import logger       # Custom logger for logging events
import json                           # For reading/writing JSON files
import joblib                         # For saving/loading binary files
from ensure import ensure_annotations # Ensures function args/returns follow type hints
from box import ConfigBox              # Converts dict to object-style access
from pathlib import Path               # Handles filesystem paths
from typing import Any                 # For generic type annotations
import base64                          # For encoding/decoding Base64 strings


# Read YAML file and return content as ConfigBox
@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    try:
        with open(path_to_yaml) as yaml_file:               # Open YAML file
            content = yaml.safe_load(yaml_file)             # Parse YAML content
            logger.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)                       # Return as object with dot-notation
    except BoxValueError:                                   # If YAML is empty
        raise ValueError("yaml file is empty")
    except Exception as e:                                  # Other exceptions
        raise e


# Create multiple directories from a list
@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)                    # Create dir if not exists
        if verbose:
            logger.info(f"created directory at: {path}")


# Save dictionary as JSON file
@ensure_annotations
def save_json(path: Path, data: dict):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)                         # Write JSON with indentation
    logger.info(f"json file saved at: {path}")


# Load JSON file and return as ConfigBox
@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    with open(path) as f:
        content = json.load(f)                               # Read JSON data
    logger.info(f"json file loaded succesfully from: {path}")
    return ConfigBox(content)                                # Access with dot-notation


# Save data as binary using joblib
@ensure_annotations
def save_bin(data: Any, path: Path):
    joblib.dump(value=data, filename=path)                   # Serialize and save
    logger.info(f"binary file saved at: {path}")


# Load binary data using joblib
@ensure_annotations
def load_bin(path: Path) -> Any:
    data = joblib.load(path)                                 # Deserialize binary file
    logger.info(f"binary file loaded from: {path}")
    return data


# Get file size in KB
@ensure_annotations
def get_size(path: Path) -> str:
    size_in_kb = round(os.path.getsize(path) / 1024)         # Calculate KB
    return f"~ {size_in_kb} KB"


# Decode Base64 string into image file
def decodeImage(imgstring, fileName):
    imgdata = base64.b64decode(imgstring)                    # Decode Base64
    with open(fileName, 'wb') as f:
        f.write(imgdata)                                     # Save as image


# Encode image file into Base64 string
def encodeImageIntoBase64(croppedImagePath):
    with open(croppedImagePath, "rb") as f:
        return base64.b64encode(f.read())                    # Return encoded bytes