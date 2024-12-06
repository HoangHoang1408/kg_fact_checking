from typing import List, Union, Dict
from transformers import AutoTokenizer
import torch
from tqdm.auto import tqdm
import pickle
import sys


class Trie:
    """A custom Trie implementation for managing token sequences."""

    def __init__(self, nested_token_ids: List[List[int]], no_subsets: bool = False):
        """
        Initialize a Trie with the given token sequences.

        Args:
            nested_token_ids: List of token ID sequences to store in the Trie
            no_subsets: If True, raises error if one sequence is a subset of another
        """
        self.max_height = (
            max(len(seq) for seq in nested_token_ids) if nested_token_ids else 0
        )
        self.trie = {
            "children": {},
            "is_end": False,
        }

        # Build trie from token sequences
        for token_ids in tqdm(nested_token_ids):
            level = self.trie["children"]
            for i, token_id in enumerate(token_ids):
                if token_id not in level:
                    level[token_id] = {"children": {}, "is_end": False}
                if i == len(token_ids) - 1:  # Mark end of sequence
                    level[token_id]["is_end"] = True
                level = level[token_id]["children"]

        # Validate no subset constraint
        if no_subsets:
            self._validate_no_subsets(nested_token_ids)

    def _validate_no_subsets(self, nested_token_ids: List[List[int]]):
        for token_ids in nested_token_ids:
            node = self.trie["children"]
            for token in token_ids:
                if token not in node:
                    break
                if node[token]["is_end"] and token != token_ids[-1]:
                    # Found a shorter sequence that is a prefix of the current sequence
                    raise ValueError(
                        f"Found a sequence that is a subset of another sequence: {token_ids}"
                    )
                node = node[token]["children"]

    def next_tokens(self, current_seq: List[int]) -> List[int]:
        """Get possible next tokens given the current sequence."""
        node = self.trie["children"]
        # Traverse to current position
        for token in current_seq:
            if token not in node:
                raise ValueError(f"Invalid sequence: {current_seq}")
            node = node[token]["children"]
        # Return possible next tokens
        return list(node.keys())

    def reached_leaf(self, current_seq: List[int]) -> bool:
        """Check if current sequence reaches a leaf node."""
        node = self.trie["children"]
        for i, token in enumerate(current_seq):
            if token not in node:
                raise ValueError(f"Sequence {current_seq} not in trie")
            if i == len(current_seq) - 1 and node[token]["is_end"]:
                return True
            node = node[token]["children"]
        return False

    def is_subset(self, candidate_seq: List[int]) -> bool:
        """Check if the given sequence is a subset (prefix) of any sequence in the trie."""
        if not candidate_seq:
            return True

        node = self.trie["children"]
        for token in candidate_seq:
            if token not in node:
                return False
            node = node[token]["children"]
        # The sequence is a subset if we can reach this point - we don't need
        # to check node["is_end"] since we're checking for prefixes
        return True

    def count_unique_paths(self):
        def _count_unique_paths(node: Dict) -> int:
            """Count unique paths in the trie."""
            count = 0
            if node["is_end"]:
                count += 1
            for token, child in node["children"].items():
                count += _count_unique_paths(child)
            return count

        return _count_unique_paths(self.trie)

    @staticmethod
    def load(filepath: str) -> "Trie":
        with open(filepath, "rb") as f:
            trie_data = pickle.load(f)
            trie = Trie([])
            trie.max_height = trie_data["max_height"]
            trie.trie = trie_data["trie"]
            return trie

    def store(self, filepath: str):
        old_limit = sys.getrecursionlimit()
        sys.setrecursionlimit(20000)

        try:
            trie_data = {"trie": self.trie, "max_height": self.max_height}
            with open(filepath, "wb") as f:
                pickle.dump(trie_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        finally:
            sys.setrecursionlimit(old_limit)


def constrained_decoding(
    tokenizer: AutoTokenizer,
    trie: Trie,
    start_entity_token: str,
    end_entity_token: str,
) -> callable:
    if not start_entity_token or not end_entity_token:
        raise ValueError("start_entity_token and end_entity_token must not be empty")

    # Get token IDs for the entity markers
    start_id = tokenizer.convert_tokens_to_ids(start_entity_token)
    end_id = tokenizer.convert_tokens_to_ids(end_entity_token)

    if start_id == tokenizer.unk_token_id or end_id == tokenizer.unk_token_id:
        raise ValueError("Failed to convert start_entity_token or end_entity_token to valid token IDs")

    all_tokens = list(range(len(tokenizer)))

    def constrained_function(batch_id: int, tokens: torch.Tensor) -> List[int]:
        """
        Apply constrained decoding based on the current token sequence.

        Args:
            batch_id: The batch index
            tokens: Current sequence of tokens (shape: [sequence_length])

        Returns:
            List of allowed next tokens that satisfy the constraints

        Note:
            Returns all possible tokens when not in entity mode or if an error occurs
        """
        tokens = tokens.tolist()
        
        # Find the last occurrence of start and end tokens
        try:
            last_start_idx = len(tokens) - 1 - tokens[::-1].index(start_id) if start_id in tokens else -1
            last_end_idx = len(tokens) - 1 - tokens[::-1].index(end_id) if end_id in tokens else -1
        except ValueError:
            return all_tokens

        # We're in entity mode if we've seen a start token more recently than an end token
        entity_mode = last_start_idx > last_end_idx

        if entity_mode:
            try:
                current_path = tokens[last_start_idx + 1:]
                next_tokens = trie.next_tokens(current_path)
                return next_tokens if next_tokens else all_tokens
            except (IndexError, ValueError) as e:
                print(f"Error in constrained decoding: {e}")
                return all_tokens
        return all_tokens

    return constrained_function
