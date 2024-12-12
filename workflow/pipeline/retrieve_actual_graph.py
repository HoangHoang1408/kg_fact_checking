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


class RetrievalUtils:
    @staticmethod
    def get_most_relevant_with_encoder(encoder, query, candidates, k=1):
        candidate_embeddings = encoder.encode(candidates)
        query_embedding = encoder.encode(query)
        scores = candidate_embeddings @ query_embedding.T
        return candidates[scores.argsort()[-k:]]


class GraphUtils:
    @staticmethod
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

    @staticmethod
    def paths_between_two_entities(
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

    @staticmethod
    def common_entities_of_list_entities(G, list_entities, max_hops=1):
        def k_hop_neighbors(G, node, k):
            return set(nx.single_source_shortest_path(G, node, k).keys())

        neighbors = set()
        for entity in list_entities:
            neighbors.update(k_hop_neighbors(G, entity, max_hops))
        return neighbors

    @staticmethod
    def retrieve_from_intermediate_graphs(G, encoder, intermediate_graphs):
        print("\n=== Starting retrieval from intermediate graphs ===")
        def retrieve_from_one_intermediate_graph(
            intermediate_graph, max_hops=1, top_k_triplets=1
        ):
            print(f"\nProcessing intermediate graph with max_hops={max_hops}, top_k_triplets={top_k_triplets}")
            def parse_triplet(triplet):
                try:
                    head, relation, tail = triplet.strip().split("||")
                    return (head.strip(), relation.strip(), tail.strip())
                except ValueError:
                    print(f"Failed to parse triplet: {triplet}")
                    return None

            def retrieve_relevant_triplets(triplet, max_hops=1, top_k_triplets=1):
                head, relation, tail = triplet
                print(f"\nFinding paths between {head} and {tail} (max_hops={max_hops})")
                candidates = GraphUtils.paths_between_two_entities(
                    G, head, tail, max_hops=max_hops
                )
                print(f"Found {len(candidates)} candidate paths")
                candidate_texts = [
                    f"{cand[0]} {cand[1]} {cand[2]}" for cand in candidates
                ]
                if len(candidate_texts) == 0:
                    print("No candidate paths found")
                    return None
                print("Getting most relevant paths using encoder...")
                results = RetrievalUtils.get_most_relevant_with_encoder(
                    encoder,
                    f"{head} {relation} {tail}",
                    candidate_texts,
                    k=top_k_triplets,
                )
                print(f"Selected top {top_k_triplets} relevant paths")
                return results

            print("\nParsing triplets from intermediate graph...")
            triplets = intermediate_graph.split("\n")
            triplets = [
                parse_triplet(triplet)
                for triplet in triplets
                if parse_triplet(triplet) is not None
            ]
            print(f"Successfully parsed {len(triplets)} triplets")
            
            retrieval_state = {
                "clarified_triplets": [],
                "unknown_triplets": [],
                "unknown_variables": {},
            }
            
            print("\nClassifying triplets...")
            for triplet in triplets:
                head, relation, tail = triplet
                if "unknown" in head or "unknown" in tail:
                    print(f"Found unknown triplet: {head} || {relation} || {tail}")
                    retrieval_state["unknown_triplets"].append(triplet)
                else:
                    print(f"Found clarified triplet: {head} || {relation} || {tail}")
                    retrieval_state["clarified_triplets"].append(
                        {
                            "triplet": triplet,
                        }
                    )

            print(f"\nProcessing {len(retrieval_state['clarified_triplets'])} clarified triplets...")
            for triplet in retrieval_state["clarified_triplets"]:
                print(f"\nRetrieving relevant triplets for: {' || '.join(triplet['triplet'])}")
                triplet["relevant_triplets"] = retrieve_relevant_triplets(
                    triplet["triplet"],
                    max_hops=max_hops,
                    top_k_triplets=top_k_triplets,
                )
                if triplet["relevant_triplets"]:
                    print("Found relevant triplets:", triplet["relevant_triplets"])
                else:
                    print("No relevant triplets found")

            # step 1: find all revelant triplets for clarified triplets


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
    kg = GraphUtils.convert_kg_to_networkx(kg)

    data = DataUtils.load_data(
        os.path.join(
            args.data_folder_path, "output", "factkg_test_with_intermediate_graph.json"
        )
    )

    for sample in tqdm(data):
        sample["retrieved_graph"] = GraphUtils.paths_between_two_entities(
            kg, sample["claim"], sample["target_entity"], args.max_retrieval_hops
        )

    # Save the updated data with retrieved paths
    output_path = os.path.join(
        args.data_folder_path, "output", "factkg_test_with_retrieved_paths.json"
    )
    DataUtils.save_data(data, output_path)
