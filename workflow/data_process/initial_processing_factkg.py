from src.utils import DataUtils
from src.utils import clean_original_entity, clean_orignal_relation
from tqdm import tqdm
from collections import defaultdict
import argparse
import os
from uuid import uuid4
import json
import random


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-folder-path", type=str, required=True)

    return parser.parse_args()


def initial_process_data(data_path):
    data = DataUtils.load_data(data_path)
    # Convert to list in one go
    data = [{"claim": k, "id": uuid4().hex, **v} for k, v in data.items()]

    # Process all entities at once
    all_entities = set()
    for sample in data:
        all_entities.update(sample["Entity_set"])

    # Create a mapping for cleaned entities
    entity_map = {
        entity: clean_original_entity(entity)
        for entity in tqdm(all_entities, desc="Cleaning unique entities")
    }

    if "Evidence" in data[0]:
        # Create a mapping for cleaned relations
        all_relations = set()
        for sample in data:
            for relations in sample["Evidence"].values():
                for relation in relations:
                    all_relations.update(relation)

        relation_map = {
            relation: clean_orignal_relation(relation)
            for relation in tqdm(all_relations, desc="Cleaning unique relations")
        }

    # Apply the mapping to all samples
    for sample in tqdm(data, desc="Applying cleaned entities"):
        sample["Entity_set"] = [entity_map[entity] for entity in sample["Entity_set"]]
        if "Evidence" in sample:
            temp = {}
            for entity, relations in sample["Evidence"].items():
                temp[entity_map[entity]] = [
                    [relation_map[r] for r in relation] for relation in relations
                ]
            sample["Evidence"] = temp

    return data


def partition_data(data, num_sample_per_partition=-1, seed=42):
    partition_types = {"num1", "multi claim", "existence", "multi hop", "negation"}
    partitions = {k: [] for k in partition_types}
    for sample in data:
        valid = False
        for t in sample["types"]:
            if t in partition_types:
                partitions[t].append(sample)
                valid = True
        if not valid:
            raise ValueError("Invalid sample")

    if num_sample_per_partition > 0:
        for k, v in partitions.items():
            random.seed(seed)
            partitions[k] = random.sample(v, num_sample_per_partition)

    return partitions


def initial_process_kg(data_path):
    kg = DataUtils.load_data(data_path)

    # Pre-process all unique entities and relations
    all_entities = set(kg.keys())
    all_relations = set()
    all_target_entities = set()

    for relations in tqdm(kg.values(), desc="Collecting unique relations and entities"):
        all_relations.update(relations.keys())
        for entities in relations.values():
            all_target_entities.update(entities)

    # Create mappings for cleaned values
    entity_map = {
        entity: clean_original_entity(entity)
        for entity in tqdm(all_entities | all_target_entities, desc="Cleaning entities")
    }
    relation_map = {
        relation: clean_orignal_relation(relation)
        for relation in tqdm(all_relations, desc="Cleaning relations")
    }

    # Build new KG using mappings
    new_kg = defaultdict(dict)
    for entity, relations in tqdm(kg.items(), desc="Building optimized KG"):
        clean_entity = entity_map[entity]
        for relation, entities in relations.items():
            clean_relation = relation_map[relation]
            new_kg[clean_entity][clean_relation] = [entity_map[e] for e in entities]

    return dict(new_kg)


if __name__ == "__main__":
    args = parse_args()

    processed_path = os.path.join(args.data_folder_path, "processed_factkg")
    if not os.path.exists(processed_path):
        os.makedirs(processed_path)

    train_data = initial_process_data(
        os.path.join(args.data_folder_path, "factkg", "factkg_train.pickle")
    )
    dev_data = initial_process_data(
        os.path.join(args.data_folder_path, "factkg", "factkg_dev.pickle")
    )
    test_data = initial_process_data(
        os.path.join(args.data_folder_path, "factkg", "factkg_test.pickle")
    )

    train_data_partitioned = partition_data(train_data, num_sample_per_partition=-1)
    dev_data_partitioned = partition_data(dev_data, num_sample_per_partition=-1)
    test_data_partitioned = partition_data(test_data, num_sample_per_partition=-1)

    train_data_partitioned_1000 = partition_data(
        train_data, num_sample_per_partition=1000
    )
    dev_data_partitioned_1000 = partition_data(dev_data, num_sample_per_partition=1000)
    test_data_partitioned_1000 = partition_data(
        test_data, num_sample_per_partition=1000
    )
    train_data_partitioned_500 = partition_data(
        train_data, num_sample_per_partition=500
    )
    dev_data_partitioned_500 = partition_data(dev_data, num_sample_per_partition=500)
    test_data_partitioned_500 = partition_data(test_data, num_sample_per_partition=500)

    os.makedirs(os.path.join(args.data_folder_path, "processed_factkg"), exist_ok=True)
    os.makedirs(
        os.path.join(args.data_folder_path, "processed_factkg", "partitioned_full"),
        exist_ok=True,
    )
    os.makedirs(
        os.path.join(args.data_folder_path, "processed_factkg", "partitioned_1000"),
        exist_ok=True,
    )
    os.makedirs(
        os.path.join(args.data_folder_path, "processed_factkg", "partitioned_500"),
        exist_ok=True,
    )

    for parition, file_name in zip(
        [train_data_partitioned, dev_data_partitioned, test_data_partitioned],
        ["train", "dev", "test"],
    ):
        DataUtils.save_json(
            parition,
            os.path.join(
                args.data_folder_path,
                "processed_factkg",
                "partitioned_full",
                f"factkg_{file_name}.json",
            ),
        )

    for parition, file_name in zip(
        [
            train_data_partitioned_1000,
            dev_data_partitioned_1000,
            test_data_partitioned_1000,
        ],
        ["train", "dev", "test"],
    ):
        DataUtils.save_json(
            parition,
            os.path.join(
                args.data_folder_path,
                "processed_factkg",
                "partitioned_1000",
                f"factkg_{file_name}.json",
            ),
        )

    for parition, file_name in zip(
        [
            train_data_partitioned_500,
            dev_data_partitioned_500,
            test_data_partitioned_500,
        ],
        ["train", "dev", "test"],
    ):
        DataUtils.save_json(
            parition,
            os.path.join(
                args.data_folder_path,
                "processed_factkg",
                "partitioned_500",
                f"factkg_{file_name}.json",
            ),
        )

    for data, file_name in zip(
        [train_data, dev_data, test_data],
        ["train", "dev", "test"],
    ):
        DataUtils.save_json(
            data,
            os.path.join(
                args.data_folder_path, "processed_factkg", f"factkg_{file_name}.json"
            ),
        )
    print("Done initial processing dataset")

    kg = initial_process_kg(
        os.path.join(args.data_folder_path, "factkg", "dbpedia_2015_undirected.pickle")
    )
    DataUtils.save_pickle(
        kg,
        os.path.join(
            args.data_folder_path, "processed_factkg", "dbpedia_2015_undirected.pkl"
        ),
    )
    kg_light = initial_process_kg(
        os.path.join(
            args.data_folder_path, "factkg", "dbpedia_2015_undirected_light.pickle"
        )
    )
    DataUtils.save_pickle(
        kg_light,
        os.path.join(
            args.data_folder_path,
            "processed_factkg",
            "dbpedia_2015_undirected_light.pkl",
        ),
    )

    print("Done initial processing knowledge graph")
