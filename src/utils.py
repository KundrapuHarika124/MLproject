import os
import sys
import dill
from src.exception import CustomException
from src.logger import logging

def save_object(file_path, obj):
    """
    Save any Python object to a file using dill (a better pickle alternative).

    Args:
        file_path (str): Full path where the object should be saved.
        obj (object): The Python object to serialize.

    Raises:
        CustomException: If any error occurs during saving.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file:
            dill.dump(obj, file)

        logging.info(f"Object successfully saved at: {file_path}")

    except Exception as e:
        logging.error(f"Error occurred while saving object to {file_path}")
        raise CustomException(e, sys)
