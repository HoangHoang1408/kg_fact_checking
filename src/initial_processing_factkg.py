from utils.load_data import DataUtil
import argparse
import os
from uuid import uuid4
import json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-folder-path", type=str, required=True)

    return parser.parse_args()


def initial_process(data_path):
    data = DataUtil.load_data(data_path)
    data = [{"claim": k, "id": uuid4().hex, **v} for k, v in data.items()]
    return data


if __name__ == "__main__":
    args = parse_args()

    train_data = initial_process(
        os.path.join(args.data_folder_path, "factkg", "factkg_train.pickle")
    )
    dev_data = initial_process(
        os.path.join(args.data_folder_path, "factkg", "factkg_dev.pickle")
    )
    test_data = initial_process(
        os.path.join(args.data_folder_path, "factkg", "factkg_test.pickle")
    )

    os.makedirs(os.path.join(args.data_folder_path, "processed_factkg"), exist_ok=True)

    DataUtil.save_json_from_list(
        train_data,
        os.path.join(args.data_folder_path, "processed_factkg", "factkg_train.json"),
    )
    DataUtil.save_json_from_list(
        dev_data,
        os.path.join(args.data_folder_path, "processed_factkg", "factkg_dev.json"),
    )
    DataUtil.save_json_from_list(
        test_data,
        os.path.join(args.data_folder_path, "processed_factkg", "factkg_test.json"),
    )
    print("Done initial processing")
