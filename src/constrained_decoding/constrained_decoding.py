from typing import List, Union, Dict
from transformers import AutoTokenizer
import torch


class Trie:
    """A custom Trie implementation for managing token sequences."""

    def __init__(self, nested_token_ids: List[List[int]], no_subsets: bool = True):
        """
        Initialize a Trie with the given token sequences.

        Args:
            nested_token_ids: List of token ID sequences to store in the Trie
            no_subsets: If True, raises error if one sequence is a subset of another
        """
        self.max_height = max(len(seq) for seq in nested_token_ids)
        self.trie = {}

        # Build trie from token sequences
        for token_ids in nested_token_ids:
            level = self.trie
            for token_id in token_ids:
                if token_id not in level:
                    level[token_id] = {}
                level = level[token_id]

        # Validate no subset constraint
        if no_subsets and self._has_subsets():
            raise ValueError(
                "Each sequence in nested_token_ids can't be a subset of another sequence, "
                f"but found such case in: {nested_token_ids}"
            )

    def next_tokens(self, current_seq: List[int]) -> List[int]:
        """Get possible next tokens given the current sequence."""
        node = self.trie
        # Traverse to current position
        for token in current_seq:
            if token not in node:
                return []
            node = node[token]
        # Return possible next tokens
        return list(node.keys())

    def reached_leaf(self, current_seq: List[int]) -> bool:
        """Check if current sequence reaches a leaf node."""
        node = self.trie
        for token in current_seq:
            if token not in node:
                raise ValueError(f"Sequence {current_seq} not in trie")
            node = node[token]

        return len(self.next_tokens(current_seq)) == 0

    def _count_leaves(self, node: Dict) -> int:
        """Count leaf nodes in the trie."""
        if not node:  # Leaf node
            return 1
        return sum(self._count_leaves(child) for child in node.values())

    def _has_subsets(self) -> bool:
        """Check if any sequence is a subset of another."""
        total_sequences = self._count_leaves(self.trie)
        return total_sequences != self._count_unique_paths(self.trie)

    def _count_unique_paths(self, node: Dict, path: tuple = ()) -> int:
        """Count unique paths in the trie."""
        if not node:  # Leaf node
            return 1
        count = 0
        for token, child in node.items():
            count += self._count_unique_paths(child, path + (token,))
        return count


def constrained_decoding(
    tokenizer: AutoTokenizer,
    trie: Trie,
    start_sequence: str,
    end_sequence: str,
) -> callable:
    """
    Create a constrained decoding function for text generation.

    Args:
        tokenizer: The tokenizer to use for encoding sequences
        trie: A Trie object containing valid token sequences
        start_sequence: The sequence marking the start of constrained generation
        end_sequence: The sequence marking the end of constrained generation

    Returns:
        A function that implements the constrained decoding logic

    Raises:
        ValueError: If start_sequence or end_sequence are empty or invalid
    """
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
        """
        Find all occurrences of subsequence in sequence.

        Args:
            sequence: The main sequence to search in
            subsequence: The subsequence to search for

        Returns:
            List of starting indices where subsequence occurs in sequence
        """
        indices = []
        if not subsequence:
            return indices
        for i in range(len(sequence) - len(subsequence) + 1):
            if sequence[i : i + len(subsequence)] == subsequence:
                indices.append(i)
        return indices

    def constrained_function(batch_id: int, tokens: torch.Tensor) -> List[int]:
        """
        Apply constrained decoding based on the current token sequence.

        Args:
            batch_id: The batch index
            tokens: Current sequence of tokens

        Returns:
            List of allowed next tokens
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
