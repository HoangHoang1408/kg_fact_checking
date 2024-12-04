from src.utils import DataUtils
from src.utils import clean_original_entity, clean_orignal_relation
from tqdm import tqdm
from collections import defaultdict
import argparse
import os
from uuid import uuid4
import json


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
                    all_relations.add(relation)

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

    os.makedirs(os.path.join(args.data_folder_path, "processed_factkg"), exist_ok=True)

    DataUtils.save_json_from_list(
        train_data,
        os.path.join(args.data_folder_path, "processed_factkg", "factkg_train.json"),
    )
    DataUtils.save_json_from_list(
        dev_data,
        os.path.join(args.data_folder_path, "processed_factkg", "factkg_dev.json"),
    )
    DataUtils.save_json_from_list(
        test_data,
        os.path.join(args.data_folder_path, "processed_factkg", "factkg_test.json"),
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
