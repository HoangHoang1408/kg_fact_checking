from src.utils import llm_generate, batch_llm_generate
from src.constrained_decoding import constrained_decoding, Trie
from src.utils import DataUtils
from argparse import ArgumentParser
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm.auto import tqdm
import os

PROMPT = """
Convert the following text into a graph format. The graph can contain unknown entities that do not exist within input text but support the input text.

Input text: {{claim}}
""".strip()


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--data-folder-path", type=str, required=True)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--num-return-sequences", type=int, default=1)
    parser.add_argument("--num-beam-groups", type=int, default=1)
    parser.add_argument("--diversity-penalty", type=float, default=1.0)
    parser.add_argument("--early-stopping", type=bool, default=True)
    parser.add_argument("--start-sequence", type=str, default="<entity>")
    parser.add_argument("--end-sequence", type=str, default="</entity>")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for processing multiple samples")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path)
    trie = Trie.load(
        os.path.join(args.data_folder_path, "processed_factkg", "entity_trie.pkl")
    )
    constrained_function = constrained_decoding(
        tokenizer, trie, args.start_sequence, args.end_sequence
    )
    generation_config = {
        "max_new_tokens": args.max_new_tokens,
        "num_beams": args.num_beams,
        "num_beam_groups": args.num_beam_groups,
        "diversity_penalty": args.diversity_penalty,
        "early_stopping": args.early_stopping,
    }
    test_data = DataUtils.load_data(
        os.path.join(args.data_folder_path, "processed_factkg", "factkg_test.json")
    )
    
    # Process data in batches
    for i in tqdm(range(0, len(test_data), args.batch_size)):
        batch = test_data[i:i + args.batch_size]
        prompts = [PROMPT.replace("{{claim}}", sample["claim"]) for sample in batch]
        
        # Generate for the entire batch
        batch_outputs = batch_llm_generate(
            input_texts=prompts,
            model=model,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            generation_config=generation_config,
            prefix_allowed_tokens_fn=constrained_function,
        )
        
        # Assign outputs back to the samples
        for sample, output in zip(batch, batch_outputs):
            sample["intermediate_graph"] = output

    output_path = os.path.join(args.data_folder_path, "output")
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    DataUtils.save_json_from_list(
        test_data,
        os.path.join(output_path, "factkg_test_with_intermediate_graph.json"),
    )
