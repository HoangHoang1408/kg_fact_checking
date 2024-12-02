from typing import List
import torch
from utils.constrained_decoding import PhrasalConstraint

def demo_beam_search_with_constraints(
    model,  # Your language model
    tokenizer,  # Your tokenizer
    input_text: str,
    required_phrase: str,
    num_beams: int = 3,
    max_length: int = 50
):
    """
    Demonstrates how PhrasalConstraint works in beam search.
    
    Args:
        model: Language model (e.g., GPT, T5, etc.)
        tokenizer: Associated tokenizer
        input_text: The input prompt
        required_phrase: Phrase that must appear in the output
        num_beams: Number of beams to maintain
        max_length: Maximum length of generated sequence
    """
    # Tokenize the required phrase
    required_tokens = tokenizer.encode(required_phrase, add_special_tokens=False)
    
    # Create the constraint
    constraint = PhrasalConstraint(required_tokens)
    
    # Initialize beam states
    class BeamState:
        def __init__(self, tokens: List[int], score: float, constraint):
            self.tokens = tokens
            self.score = score
            self.constraint = constraint.copy(stateful=True)
    
    # Tokenize input
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    
    # Initialize beams with the input
    beams = [BeamState(input_ids[0].tolist(), 0.0, constraint)]
    
    # Track if we found at least one completion
    found_completion = False
    
    # Beam search loop
    for step in range(max_length):
        all_candidates = []
        
        # Generate candidates for each beam
        for beam in beams:
            # Get model predictions
            with torch.no_grad():
                outputs = model(torch.tensor([beam.tokens]))
                next_token_logits = outputs.logits[0, -1, :]
                next_token_probs = torch.softmax(next_token_logits, dim=-1)
            
            # Get top k candidates
            topk_probs, topk_tokens = torch.topk(next_token_probs, k=num_beams)
            
            for prob, token in zip(topk_probs, topk_tokens):
                token = token.item()
                
                # If constraint is already completed, accept any token
                if beam.constraint.completed:
                    new_tokens = beam.tokens + [token]
                    new_score = beam.score + torch.log(prob).item()
                    new_constraint = beam.constraint.copy(stateful=True)
                    all_candidates.append(BeamState(new_tokens, new_score, new_constraint))
                    continue
                
                # Check if this token advances the constraint
                stepped, completed, reset = beam.constraint.update(token)
                
                if reset and not found_completion:
                    # Token breaks the constraint and we haven't found a completion yet - skip
                    continue
                
                # Create new beam state
                new_tokens = beam.tokens + [token]
                new_score = beam.score + torch.log(prob).item()
                new_constraint = beam.constraint.copy(stateful=True)
                
                # If this completes the constraint, mark it
                if completed:
                    found_completion = True
                    # Give a bonus score for completing the constraint
                    new_score += 50  # Reward for constraint completion
                
                all_candidates.append(BeamState(new_tokens, new_score, new_constraint))
        
        if not all_candidates and not found_completion:
            # No valid candidates found and we haven't completed constraint yet
            # Force using constraint's next token
            for beam in beams:
                if not beam.constraint.completed:
                    next_required = beam.constraint.advance()
                    if next_required is not None:
                        new_tokens = beam.tokens + [next_required]
                        # Use a high negative score to penalize forced tokens
                        new_score = beam.score - 100
                        new_constraint = beam.constraint.copy(stateful=True)
                        new_constraint.update(next_required)
                        all_candidates.append(BeamState(new_tokens, new_score, new_constraint))
        
        # Select top beams for next step
        beams = sorted(all_candidates, key=lambda x: x.score, reverse=True)[:num_beams]
        
        # Check if we've reached max length
        if len(beams[0].tokens) >= max_length:
            break
        
        # Optional: Check if we should stop (e.g., all beams generated an end token)
        # if all(tokenizer.eos_token_id in beam.tokens for beam in beams):
        #     break
    
    # Return the best completion
    best_beam = max(beams, key=lambda x: x.score)
    return tokenizer.decode(best_beam.tokens)

# Example usage:
"""
# Assuming you have a model and tokenizer:
model = ...  # Your language model
tokenizer = ...  # Your tokenizer

input_text = "Write a story about"
required_phrase = "a brave knight"

output = demo_beam_search_with_constraints(
    model=model,
    tokenizer=tokenizer,
    input_text=input_text,
    required_phrase=required_phrase
)
print(output)
"""
