"""
Utility Module for Model and File Management
--------------------------------------------
This module provides essential utility functions and constants for managing model paths
and file listing in the automated grading system project. It includes an `Enum` class for
model paths and a file listing function that helps retrieve data files from specified directories.

Author: Anand Kamble
"""

import os
from enum import Enum

# ############## CONSTANTS ################
MODEL_DIR = "./"


class Models(Enum):
    """
    Enum for storing model file paths.
    
    Attributes:
    ----------
    MISTRAL : str
        Path to the Mistral model file.
    """
    MISTRAL = MODEL_DIR + "mistral-7b-instruct-v0.1.Q3_K_L.gguf"


def list_files(directory: Models) -> list:
    """
    List all files in the specified directory.

    Args:
    -----
    directory : Models
        The directory Enum value representing the path to list files from.

    Returns:
    -------
    list : list of str
        A list of file names found in the directory.
    """
    files = []
    for file in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, file)):
            files.append(file)
    return files
