from typing import List, Dict, Tuple, Set
import heapq
import math
from custom_constraint import CustomConstraint, CustomTrie

class BeamState:
    """Represents a state in beam search."""
    def __init__(self, tokens: List[int], score: float, constraint: CustomConstraint):
        self.tokens = tokens
        self.score = score
        self.constraint = constraint
        
    def __lt__(self, other):
        # For heap comparison - higher scores should come first
        return self.score > other.score

class SimpleGraphLM:
    """A simple graph-based language model with constrained generation."""
    
    def __init__(self):
        # Initialize vocabulary
        self.vocab = {
            "<s>": 0,      # Start token
            "</s>": 1,     # End token
            "<entity>": 2, # Entity start token
            "</entity>": 3,# Entity end token
            "Tom": 4,
            "has": 5,
            "a": 6,
            "house": 7,
            "with": 8,
            "three": 9,
            "four": 10,
            "floors": 11,
            "||": 12,      # Relation separator
            ".": 13,
            "\n": 14,
        }
        self.id2token = {v: k for k, v in self.vocab.items()}
        
        # Define transition probabilities
        self.transitions = {
            0: {4: 0.5, 6: 0.5},  # <s> -> Tom/a
            4: {5: 0.8, 12: 0.2},  # Tom -> has/||
            5: {6: 1.0},  # has -> a
            6: {7: 1.0},  # a -> house
            7: {8: 0.4, 12: 0.6},  # house -> with/||
            8: {9: 1.0},  # with -> three
            9: {10: 1.0},  # three -> floors
            10: {13: 0.5, 12: 0.5},  # floors -> ./||
            12: {2: 0.5, 5: 0.25, 8: 0.25},  # || -> <entity>/has/with
            2: {4: 0.3, 7: 0.4, 9: 0.3},  # <entity> -> Tom/house/three
            3: {12: 0.6, 13: 0.4},  # </entity> -> ||/.
            13: {14: 1.0},  # . -> \n
            14: {1: 1.0},  # \n -> </s>
        }
        
        # Define valid entity sequences
        self.entity_sequences = [
            [self.vocab["Tom"]],
            [self.vocab["house"]],
            [self.vocab["floors"]],
        ]
        
        # Initialize constraint
        self.base_constraint = CustomConstraint(
            self.entity_sequences,
            self.vocab["<entity>"],
            self.vocab["</entity>"]
        )
    
    def get_next_tokens(self, current_token: int) -> Dict[int, float]:
        """Get possible next tokens and their probabilities."""
        return self.transitions.get(current_token, {})
    
    def beam_search(self, beam_size: int = 3, max_length: int = 30) -> List[Tuple[List[int], float]]:
        """Perform beam search with constrained generation."""
        # Initialize beam with start token
        start_state = BeamState(
            tokens=[self.vocab["<s>"]],
            score=0.0,
            constraint=self.base_constraint.copy(stateful=True)
        )
        beam = [start_state]
        
        finished_sequences = []
        
        for _ in range(max_length):
            next_beam = []
            
            # Expand each sequence in the beam
            for state in beam:
                current_token = state.tokens[-1]
                
                # Get possible next tokens from both LM and constraint
                lm_next_tokens = self.get_next_tokens(current_token)
                constraint_next_tokens = state.constraint.advance()
                
                # Filter tokens based on constraint
                valid_next_tokens = {}
                for token_id, prob in lm_next_tokens.items():
                    if constraint_next_tokens is None or token_id in constraint_next_tokens:
                        if state.constraint.does_advance(token_id):
                            valid_next_tokens[token_id] = prob
                
                # Add candidates to next beam
                for token_id, prob in valid_next_tokens.items():
                    # Create new constraint state
                    new_constraint = state.constraint.copy(stateful=True)
                    stepped, completed, reset = new_constraint.update(token_id)
                    
                    if reset:
                        continue
                    
                    new_tokens = state.tokens + [token_id]
                    new_score = state.score + math.log(prob)
                    
                    if token_id == self.vocab["</s>"]:
                        finished_sequences.append((new_tokens, new_score))
                    else:
                        next_beam.append(BeamState(new_tokens, new_score, new_constraint))
            
            # Keep top beam_size sequences
            next_beam = heapq.nlargest(beam_size, next_beam, key=lambda x: x.score)
            
            if not next_beam:
                break
                
            beam = next_beam
        
        # Add any unfinished sequences to results
        for state in beam:
            finished_sequences.append((state.tokens, state.score))
        
        return sorted(finished_sequences, key=lambda x: x[1], reverse=True)
    
    def tokens_to_text(self, tokens: List[int]) -> str:
        """Convert token IDs to readable text."""
        return " ".join(self.id2token[t] for t in tokens)

def main():
    """Test the constrained language model generation."""
    print("Initializing Graph Language Model...")
    model = SimpleGraphLM()
    
    print("\nGenerating sequences with beam search...")
    beam_size = 3
    sequences = model.beam_search(beam_size=beam_size)
    
    print(f"\nTop {len(sequences)} sequences generated:")
    for i, (tokens, score) in enumerate(sequences):
        print(f"\nSequence {i+1} (score: {score:.3f}):")
        text = model.tokens_to_text(tokens)
        print(text)

if __name__ == "__main__":
    main()