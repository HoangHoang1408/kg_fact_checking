from transformers import (
    AutoModelForCausalLM,
    PreTrainedTokenizer,
)
from typing import List

DEFAULT_GENERATION_CONFIG = {
    "max_new_tokens": 128,
    "do_sample": False,
    "num_beams": 5,
    "num_beam_groups": 1,
    "diversity_penalty": 0.0,
    "temperature": 0.0,
    "early_stopping": True,
}


def llm_generate(
    input_text: str,
    model: AutoModelForCausalLM,
    tokenizer: PreTrainedTokenizer,
    generation_config: dict = DEFAULT_GENERATION_CONFIG,
    prefix_allowed_tokens_fn=None,
    **kwargs,
) -> List[str]:
    """
    Generate text using a Hugging Face model.

    Args:
        input_text (str): The input text to generate based on.
        model (AutoModelForCausalLM): The Hugging Face model to use for generation.
        tokenizer (AutoTokenizer): The tokenizer to use for encoding the input and decoding the output.
        generation_config (dict, optional): A config dictionary for the `generate` method of the model. Defaults to DEFAULT_GENERATION_CONFIG.
        **kwargs: Additional keyword arguments passed to the `generate` method of the model.

    Returns:
        List[str]: The generated text as a list of strings.
    """
    messages = [{"role": "user", "content": input_text}]
    input_text = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    inputs = tokenizer(input_text, return_tensors="pt", add_special_tokens=False).to(
        model.device
    )
    generation_config = {
        **generation_config,
        **kwargs,
        "pad_token_id": tokenizer.eos_token_id or tokenizer.pad_token_id,
        "eos_token_id": [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ],
    }
    outputs = model.generate(
        **inputs,
        **generation_config,
        output_scores=True,
        prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
    )
    sequences = outputs.cpu()[:, len(inputs["input_ids"][0]) :]
    return tokenizer.batch_decode(sequences, skip_special_tokens=True)


def batch_llm_generate(
    input_texts: List[str],
    model: AutoModelForCausalLM,
    tokenizer: PreTrainedTokenizer,
    batch_size: int = 8,
    generation_config: dict = DEFAULT_GENERATION_CONFIG,
    prefix_allowed_tokens_fn: callable = None,
    **kwargs,
) -> List[str]:
    """
    Generate text for multiple inputs in batches using a Hugging Face model.

    Args:
        input_texts (List[str]): List of input texts to generate based on.
        model (AutoModelForCausalLM): The Hugging Face model to use for generation.
        tokenizer (AutoTokenizer): The tokenizer to use for encoding the input and decoding the output.
        batch_size (int, optional): Size of batches to process at once. Defaults to 8.
        generation_config (dict, optional): A config dictionary for the `generate` method of the model.
        prefix_allowed_tokens_fn (callable, optional): Function that filters allowed tokens at each step.
        **kwargs: Additional keyword arguments passed to the `generate` method of the model.

    Returns:
        List[str]: The generated texts as a list of strings, in the same order as input_texts.
    """
    all_outputs = []

    # Process in batches
    for i in range(0, len(input_texts), batch_size):
        batch_texts = input_texts[i : i + batch_size]

        # Convert texts to chat format and tokenize
        messages_batch = [[{"role": "user", "content": text}] for text in batch_texts]
        formatted_texts = [
            tokenizer.apply_chat_template(
                msgs, add_generation_prompt=True, tokenize=False
            )
            for msgs in messages_batch
        ]

        # Tokenize all inputs in the batch
        inputs = tokenizer(
            formatted_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=False,
            padding_side="left",
        ).to(model.device)

        # Set up generation config
        gen_config = {
            **generation_config,
            **kwargs,
            "pad_token_id": tokenizer.eos_token_id or tokenizer.pad_token_id,
            "eos_token_id": [
                tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids("<|eot_id|>"),
            ],
        }

        # Generate for the batch
        outputs = model.generate(
            **inputs,
            **gen_config,
            output_scores=True,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        )

        # Process outputs
        input_length = inputs["input_ids"].shape[1]
        sequences = outputs.cpu()[:, input_length:]
        decoded_outputs = tokenizer.batch_decode(sequences, skip_special_tokens=True)

        # group decoded outputs by batch size
        decoded_outputs = [
            decoded_outputs[i : (i + 1) * gen_config["num_return_sequences"]]
            for i in range(len(decoded_outputs) // gen_config["num_return_sequences"])
        ]
        all_outputs.extend(decoded_outputs)

    return all_outputs
