from typing import List, Union, Dict
from transformers import AutoTokenizer
import torch
from tqdm.auto import tqdm
import pickle

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
    def load(filepath: str) -> 'Trie':
        """
        Load a Trie object from a file.
        
        Args:
            filepath: Path to the file containing the serialized Trie
            
        Returns:
            Loaded Trie object
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            pickle.UnpicklingError: If there's an error during deserialization
        """
        with open(filepath, 'rb') as f:
            trie_data = pickle.load(f)
            # Create an empty Trie
            trie = Trie([])
            # Update its attributes
            trie.trie = trie_data['trie']
            trie.max_height = trie_data['max_height']
            return trie

    def store(self, filepath: str):
        """
        Store the Trie object to a file using pickle serialization.
        
        Args:
            filepath: Path where to save the serialized Trie
            
        Raises:
            IOError: If there's an error writing to the file
        """
        trie_data = {
            'trie': self.trie,
            'max_height': self.max_height
        }
        with open(filepath, 'wb') as f:
            pickle.dump(trie_data, f, protocol=pickle.HIGHEST_PROTOCOL)


def constrained_decoding(
    tokenizer: AutoTokenizer,
    trie: Trie,
    start_sequence: str,
    end_sequence: str,
) -> callable:
    if not start_sequence or not end_sequence:
        raise ValueError("start_sequence and end_sequence must not be empty")

    # Tokenize sequences
    start_tokens = tokenizer(start_sequence, return_tensors="pt")
    end_tokens = tokenizer(end_sequence, return_tensors="pt")

    if not start_tokens.input_ids.nelement() or not end_tokens.input_ids.nelement():
        raise ValueError("Failed to tokenize start_sequence or end_sequence")

    start_ids = start_tokens.input_ids[0].tolist()
    end_ids = end_tokens.input_ids[0].tolist()
    all_tokens = list(range(len(tokenizer)))

    def check_list_contain(sequence: List[int], subsequence: List[int]) -> List[int]:
        indices = []
        if not subsequence:
            return indices

        subseq_len = len(subsequence)
        first_token = subsequence[0]

        i = 0
        while i <= len(sequence) - subseq_len:
            if sequence[i] == first_token:
                if sequence[i : i + subseq_len] == subsequence:
                    indices.append(i)
                i += 1
            else:
                i += 1

        return indices

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
        all_start_indices = check_list_contain(tokens, start_ids)
        all_end_indices = check_list_contain(tokens, end_ids)

        # Handle cases where start or end sequences are not found
        if not all_start_indices:
            return all_tokens
        if not all_end_indices:
            all_end_indices = [-1]  # Use -1 to ensure entity_mode is True

        candidate_start_index = all_start_indices[-1]
        candidate_end_index = all_end_indices[-1]

        entity_mode = candidate_start_index > candidate_end_index

        if entity_mode:
            try:
                current_path = tokens[candidate_start_index:]
                next_tokens = trie.next_tokens(current_path)
                return next_tokens if next_tokens else all_tokens
            except (IndexError, ValueError) as e:
                print(f"Error in constrained decoding: {e}")
                return all_tokens
        return all_tokens

    return constrained_function
