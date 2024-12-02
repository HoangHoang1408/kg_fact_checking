from tqdm.auto import tqdm
import json
import pickle
from datasets import load_from_disk


class LoadData:
    """
    A utility class for loading data from different file formats.

    Methods:

        load_data:
            Args:
                file_path (str): The path to the file to load.
            Returns:
                data (list or dict or Dataset or None): The loaded data.
    """

    @staticmethod
    def load_data(file_path: str) -> list | dict | None:
        """
        Loads data from a file.

        Args:
            file_path (str): The path to the file to load.

        Returns:
            data (list or dict or None): The loaded data.
        """
        data = None
        if file_path.endswith(".jsonl"):
            with open(file_path, "r") as file:
                data = [json.loads(line) for line in file]
        elif file_path.endswith(".json"):
            with open(file_path, "r") as file:
                data = json.load(file)
        elif file_path.endswith(".pickle"):
            with open(file_path, "rb") as file:
                data = pickle.load(file)
        elif file_path.endswith(".txt"):
            with open(file_path, "r") as file:
                data = [line.strip() for line in file]
        else:
            data = load_from_disk(file_path)
        if data is None:
            raise ValueError(f"File {file_path} is not a valid format.")
        return data
