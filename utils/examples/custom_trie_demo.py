from typing import List
from utils.custom_constraint import CustomTrie

def print_section(title: str):
    """Helper function to print formatted section titles."""
    print("\n" + "=" * 50)
    print(title)
    print("=" * 50)

def demonstrate_basic_trie():
    """Demonstrate basic Trie functionality with simple token sequences."""
    print_section("Basic Trie Operations")
    
    # Create a simple trie with token sequences
    sequences = [
        [1, 2, 3],  # Sequence 1: 1->2->3
        [1, 2, 4],  # Sequence 2: 1->2->4
        [5, 6]      # Sequence 3: 5->6
    ]
    
    print("Creating Trie with sequences:", sequences)
    trie = CustomTrie(sequences)
    
    # Demonstrate next_tokens functionality
    print("\nDemonstrating next_tokens:")
    print("Possible first tokens:", trie.next_tokens([]))  # Should return [1, 5]
    print("After token 1:", trie.next_tokens([1]))        # Should return [2]
    print("After 1,2:", trie.next_tokens([1, 2]))        # Should return [3, 4]
    print("After 5:", trie.next_tokens([5]))             # Should return [6]
    print("After invalid sequence:", trie.next_tokens([9]))  # Should return []

def demonstrate_leaf_detection():
    """Demonstrate leaf node detection functionality."""
    print_section("Leaf Node Detection")
    
    sequences = [[1, 2, 3], [4, 5]]
    trie = CustomTrie(sequences)
    
    print("Sequences:", sequences)
    print("\nChecking if sequences reach leaf nodes:")
    print("[1, 2, 3] reaches leaf:", trie.reached_leaf([1, 2, 3]))      # True
    print("[1, 2] reaches leaf:", trie.reached_leaf([1, 2]))            # False
    print("[4, 5] reaches leaf:", trie.reached_leaf([4, 5]))            # True
    print("[4] reaches leaf:", trie.reached_leaf([4]))                  # False

def demonstrate_subset_validation():
    """Demonstrate subset validation functionality."""
    print_section("Subset Validation")
    
    print("Attempting to create Trie with subset sequences:")
    try:
        # This should raise an error because [1, 2] is a subset of [1, 2, 3]
        sequences = [[1, 2], [1, 2, 3]]
        trie = CustomTrie(sequences)
    except ValueError as e:
        print("Caught expected error:", str(e))
    
    print("\nCreating valid Trie without subsets:")
    sequences = [[1, 2, 3], [1, 2, 4], [1, 3, 4]]
    trie = CustomTrie(sequences)
    print("Successfully created Trie with sequences:", sequences)

def demonstrate_entity_sequences():
    """Demonstrate Trie usage with entity-like sequences."""
    print_section("Entity Sequence Example")
    
    # Example token IDs for entities (in practice, these would come from your tokenizer)
    entity_sequences = [
        [10, 20, 30],  # Example: "New York"
        [10, 25],      # Example: "New Jersey"
        [40, 50, 60]   # Example: "San Francisco"
    ]
    
    print("Entity sequences:", entity_sequences)
    trie = CustomTrie(entity_sequences)
    
    # Demonstrate entity completion
    print("\nEntity completion examples:")
    current = [10]  # Starting with token for "New"
    while current:
        print(f"Current sequence {current}, next possible tokens: {trie.next_tokens(current)}")
        if trie.reached_leaf(current):
            print(f"Completed entity sequence: {current}")
            break
        # For demonstration, we'll take the first available token
        next_tokens = trie.next_tokens(current)
        if next_tokens:
            current.append(next_tokens[0])
        else:
            break

def main():
    """Run all demonstrations."""
    demonstrate_basic_trie()
    demonstrate_leaf_detection()
    demonstrate_subset_validation()
    demonstrate_entity_sequences()

if __name__ == "__main__":
    main()
