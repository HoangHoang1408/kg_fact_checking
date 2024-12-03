from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

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
    tokenizer: AutoTokenizer,
    generation_config: dict = DEFAULT_GENERATION_CONFIG,
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
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    generation_config = {
        **generation_config,
        **kwargs,
        "pad_token_id": tokenizer.eos_token_id or tokenizer.pad_token_id,
    }
    outputs = model.generate(
        **inputs, **generation_config, output_scores=True, return_dict_in_generate=True
    )
    sequences = outputs.sequences.cpu()[:, len(inputs["input_ids"][0]) :]
    return tokenizer.batch_decode(sequences, skip_special_tokens=True)
