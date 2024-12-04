# from intermediate graph generated from generate_intermediate_graph.py
# expand the actual graph

from src.utils import DataUtils
import os
import argparse
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm
import networkx as nx
from typing import List, Tuple


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-folder-path", type=str, required=True)
    parser.add_argument("--encoder-path", type=str, required=True)
    parser.add_argument(
        "--graph-version",
        type=str,
        default="light",
        choices=["light", "full"],
    )
    parser.add_argument("--max-retrieval-hops", type=int, default=1)

    return parser.parse_args()


def get_most_relevant(encoder, query, candidates, k=1):
    candidate_embeddings = encoder.encode(candidates)
    query_embedding = encoder.encode(query)
    scores = candidate_embeddings @ query_embedding.T
    return candidates[scores.argsort()[-k:]]


def graph_shortest_paths(
    graph: nx.Graph, source: str, target: str, max_hops: int
) -> List[List[Tuple[str, str, str]]]:
    """
    Find all paths between source and target nodes within max_hops in the graph.

    Args:
        graph: NetworkX graph object
        source: Source node
        target: Target node
        max_hops: Maximum number of hops allowed

    Returns:
        List of paths, where each path is a list of (node1, edge_label, node2) tuples
    """
    # Check if both nodes exist in the graph
    if not graph.has_node(source):
        return []
    if not graph.has_node(target):
        return []

    all_paths = []

    # Use NetworkX's simple_paths to get all paths within max_hops
    for path in nx.all_simple_paths(
        graph, source=source, target=target, cutoff=max_hops
    ):
        # Convert path nodes to edge sequence with labels
        path_with_edges = []
        for i in range(len(path) - 1):
            node1, node2 = path[i], path[i + 1]
            # Get edge data (assuming edge labels are stored in 'label' attribute)
            edge_data = graph.get_edge_data(node1, node2)
            edge_label = edge_data.get("label", "")
            path_with_edges.append((node1, edge_label, node2))
        all_paths.append(path_with_edges)

    return all_paths


def convert_kg_to_networkx(kg: dict) -> nx.Graph:
    """
    Convert knowledge graph dictionary to NetworkX graph.

    Args:
        kg: Dictionary with format {entity1: {relation1: [entity2, entity3], ...}, ...}

    Returns:
        NetworkX undirected graph with nodes as entities and edges with relation labels
    """
    G = nx.Graph()

    # Iterate through each source entity and its relations
    for source_entity, relations in kg.items():
        # Add the source entity as a node if it doesn't exist
        if not G.has_node(source_entity):
            G.add_node(source_entity)

        # Iterate through each relation and its target entities
        for relation, target_entities in relations.items():
            # Add edges between source and all target entities with the relation as label
            for target_entity in target_entities:
                if not G.has_node(target_entity):
                    G.add_node(target_entity)
                G.add_edge(source_entity, target_entity, label=relation)

    return G


def retrieve_algorithm(G, encoder, intermediate_graph):
    triplets = intermediate_graph.split("\n")
    triplets = [
        (triplet.split("||")[0], triplet.split("||")[1], triplet.split("||")[2])
        for triplet in triplets
    ]
    unknown_entity_map = {}


if __name__ == "__main__":
    args = parse_args()
    encoder = SentenceTransformer(args.encoder_path)
    kg = DataUtils.load_data(
        os.path.join(
            args.data_folder_path,
            "processed_factkg",
            (
                "dbpedia_2015_undirected_light.pickle"
                if args.graph_version == "light"
                else "dbpedia_2015_undirected.pickle"
            ),
        )
    )
    kg = convert_kg_to_networkx(kg)

    data = DataUtils.load_data(
        os.path.join(
            args.data_folder_path, "output", "factkg_test_with_intermediate_graph.json"
        )
    )

    for sample in tqdm(data):
        sample["retrieved_graph"] = graph_shortest_paths(
            kg, sample["claim"], sample["target_entity"], args.max_retrieval_hops
        )
    # Save the updated data with retrieved paths
    output_path = os.path.join(
        args.data_folder_path, "output", "factkg_test_with_retrieved_paths.json"
    )
    DataUtils.save_data(data, output_path)
