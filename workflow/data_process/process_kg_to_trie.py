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
    # Convert set to list for tqdm
    entities_list = list(entities)
    print(f"Tokenizing {len(entities_list)} entities...")

    # Process entities in batches for memory efficiency
    batch_size = 1000
    token_sequences = []

    for i in tqdm(range(0, len(entities_list), batch_size), desc="Tokenizing entities"):
        batch = entities_list[i : i + batch_size]
        batch_tokens = tokenizer(batch, add_special_tokens=False).input_ids
        token_sequences.extend(batch_tokens)

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
        "--end-entity-token",
        type=str,
        default="</entity>",
        help="End entity token for constrained decoding",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    # Convert end entity token to ID
    end_token_id = tokenizer.convert_tokens_to_ids(args.end_entity_token)
    if end_token_id == tokenizer.unk_token_id:
        raise ValueError(
            f"End entity token '{args.end_entity_token}' is not in the tokenizer vocabulary"
        )

    kg = DataUtils.load_data(args.kg_path)

    entities = extract_entities(kg)
    print(f"Found {len(entities)} unique entities")

    print("Converting entities to token sequences")
    token_sequences = tokenize_entities(entities, tokenizer)
    print("Adding end entity token")
    token_sequences = [
        token_sequence + [end_token_id]
        for token_sequence in tqdm(token_sequences, desc="Adding end token")
    ]

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
