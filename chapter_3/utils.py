from enum import Enum

# ############## CONSTANTS ################
MODEL_DIR = "/scratch/data_amk23j/"


class Models(Enum):
    MISTRAL = MODEL_DIR + "mistral-7b-instruct-v0.1.Q3_K_L.gguf"


""" This module contains utility functions for the project. """
import os


def list_files(directory: Models):
    """
    List all files in the specified directory.

    Args:
        directory (Model): The path to the directory.

    Returns:
        list: A list of file names in the directory.
    """
    files = []
    for file in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, file)):
            files.append(file)
    return files
