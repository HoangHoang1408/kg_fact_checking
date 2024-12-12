from tqdm.auto import tqdm
import json
import pickle
from datasets import load_from_disk, Dataset


class DataUtils:
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
        elif file_path.endswith(".pickle") or file_path.endswith(".pkl"):
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

    @staticmethod
    def save_json(data, file_path):
        """
        Save a list of dictionaries to a JSON file.

        Args:
            data (list): A list of dictionaries to be saved.
            file_path (str): The path to the JSON file to be created.
        """
        with open(file_path, "w") as f:
            json.dump(data, f, ensure_ascii=False)

    @staticmethod
    def save_jsonl_from_list(data, file_path):
        """
        Save a list of dictionaries to a JSONL file.

        Args:
            data (list): A list of dictionaries to be saved.
            file_path (str): The path to the JSONL file to be created.
        """
        Dataset.from_list(data).to_json(file_path, force_ascii=False)

    @staticmethod
    def save_pickle(data, file_path):
        with open(file_path, "wb") as f:
            pickle.dump(data, f)
