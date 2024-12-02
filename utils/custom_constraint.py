from typing import List, Union, Dict

class CustomTrie:
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


class CustomConstraint:
    """
    A custom constraint for graph language model generation that:
    1. Allows free generation by default
    2. When '<entity>' is encountered, switches to constrained mode using a Trie
    3. When '</entity>' is encountered, switches back to free generation mode

    Args:
        entity_token_ids (List[List[int]]): List of valid token sequences that can appear between <entity> tags
        entity_start_token_id (int): Token ID for '<entity>'
        entity_end_token_id (int): Token ID for '</entity>'
    """

    def __init__(self, entity_token_ids: List[List[int]], entity_start_token_id: int, entity_end_token_id: int):
        # Initialize the trie with valid entity token sequences
        self.trie = CustomTrie(entity_token_ids)
        self.entity_start_token_id = entity_start_token_id
        self.entity_end_token_id = entity_end_token_id
        
        # State tracking
        self.in_entity = False  # Whether we're currently inside an entity tag
        self.current_entity_tokens = []  # Tokens collected for current entity
        self.completed = False
        
        # Store original token sequences for copying
        self.entity_token_ids = entity_token_ids

    def advance(self) -> Union[int, List[int], None]:
        """Returns the next possible token(s) based on the current state."""
        if not self.in_entity:
            # In free generation mode, no specific token is required
            return None
        else:
            # In entity mode, get next possible tokens from trie
            return self.trie.next_tokens(self.current_entity_tokens)

    def does_advance(self, token_id: int) -> bool:
        """Checks if the token advances the constraint."""
        if not self.in_entity:
            # In free mode, any token advances except entity_start requires switching modes
            return token_id != self.entity_start_token_id
        else:
            # In entity mode, must be either a valid next token or the end tag
            next_tokens = self.trie.next_tokens(self.current_entity_tokens)
            return token_id in next_tokens or token_id == self.entity_end_token_id

    def update(self, token_id: int) -> tuple[bool, bool, bool]:
        """Updates constraint state based on the token."""
        if not isinstance(token_id, int):
            raise TypeError(f"`token_id` has to be an `int`, but is {token_id} of type {type(token_id)}")

        stepped = False
        completed = False
        reset = False

        if not self.in_entity:
            if token_id == self.entity_start_token_id:
                # Entering entity mode
                self.in_entity = True
                self.current_entity_tokens = []
                stepped = True
            else:
                # Normal free generation
                stepped = True
        else:
            if token_id == self.entity_end_token_id:
                # Exiting entity mode
                if len(self.current_entity_tokens) == 0:
                    # Empty entity is not allowed
                    reset = True
                    self.reset()
                else:
                    # Successfully completed an entity
                    self.in_entity = False
                    self.current_entity_tokens = []
                    stepped = True
            else:
                next_tokens = self.trie.next_tokens(self.current_entity_tokens)
                if token_id in next_tokens:
                    # Valid entity token
                    self.current_entity_tokens.append(token_id)
                    stepped = True
                else:
                    # Invalid token in entity mode
                    reset = True
                    self.reset()

        return stepped, completed, reset

    def reset(self) -> None:
        """Resets to initial state."""
        self.in_entity = False
        self.current_entity_tokens = []
        self.completed = False

    def remaining(self) -> int:
        """Returns remaining steps needed.
        Since this is a stateful constraint that can go on indefinitely,
        we return 0 if not in entity mode (free generation)
        or 1 if in entity mode (need to complete current entity)
        """
        return 1 if self.in_entity else 0

    def copy(self, stateful: bool = False) -> 'CustomConstraint':
        """Creates a new instance of this constraint."""
        new_constraint = CustomConstraint(
            self.entity_token_ids,
            self.entity_start_token_id,
            self.entity_end_token_id
        )

        if stateful:
            new_constraint.in_entity = self.in_entity
            new_constraint.current_entity_tokens = self.current_entity_tokens.copy()
            new_constraint.completed = self.completed

        return new_constraint
