import argparse
import os
import pickle
from typing import Dict, List, Set

from src.constrained_decoding.constrained_decoding import Trie
from src.utils import DataUtils
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizer


def extract_entities(kg: Dict) -> Set[str]:
    """Extract all unique entities from the knowledge graph."""
    entities = set()

    # Add head entities
    entities.update(kg.keys())

    # Add tail entities
    for relations in tqdm(kg.values()):
        temp_entities = []
        for relation in relations.values():
            temp_entities.extend(relation)
        entities.update(temp_entities)

    return entities


def tokenize_entities(
    entities: Set[str], tokenizer: PreTrainedTokenizer
) -> List[List[int]]:
    """Convert entities to token IDs using batch encoding for improved efficiency."""
    # Batch encode all entities at once
    token_sequences = tokenizer(list(entities), add_special_tokens=False).input_ids
    token_sequences = [
        token_sequence for token_sequence in token_sequences if token_sequence
    ]
    return token_sequences


def parse_args():
    parser = argparse.ArgumentParser(description="Process KG to Trie")
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        required=True,
        help="Path to the pretrained tokenizer",
    )
    parser.add_argument(
        "--kg-path",
        type=str,
        required=True,
        help="Path to the knowledge graph pickle file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory to save the Trie",
    )
    parser.add_argument(
        "--end-sequence",
        type=str,
        default="</entity>",
        help="End sequence for constrained decoding",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    end_ids = tokenizer.encode(args.end_sequence, add_special_tokens=False)
    kg = DataUtils.load_data(args.kg_path)

    entities = extract_entities(kg)
    print(f"Found {len(entities)} unique entities")

    print("Converting entities to token sequences")
    token_sequences = tokenize_entities(entities, tokenizer)
    token_sequences = [token_sequence + end_ids for token_sequence in token_sequences]

    print("Creating Trie from token sequences")
    trie = Trie(token_sequences, no_subsets=False)

    print(f"Saving Trie to {args.output_dir}")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    output_path = os.path.join(args.output_dir, "entity_trie.pkl")
    trie.store(output_path)

    print("Done!")


if __name__ == "__main__":
    main()
