from src.constrained_decoding import constrained_decoding, Trie
from src.utils import batch_llm_generate, DataUtils
from argparse import ArgumentParser
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm.auto import tqdm
import os
import logging
from src.prompts import GEN_PSEUDO_GRAPH_PROMPT

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--partition-file-path", type=str, required=True)
    parser.add_argument("--trie-path", type=str, required=True)
    parser.add_argument("--output-folder-path", type=str, required=True)
    parser.add_argument("--version", type=str, required=True, default="1.0")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--num-return-sequences", type=int, default=1)
    parser.add_argument("--num-beam-groups", type=int, default=1)
    parser.add_argument("--diversity-penalty", type=float, default=1.0)
    parser.add_argument("--early-stopping", type=bool, default=True)
    parser.add_argument("--start-token", type=str, default="<entity>")
    parser.add_argument("--end-token", type=str, default="</entity>")
    parser.add_argument(
        "--partition",
        type=str,
        choices=["existence", "num1", "multi claim", "multi hop", "negation"],
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for processing multiple samples",
    )
    parser.add_argument("--use-constrained-decoding", type=bool, default=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    logger.info(f"Starting graph generation for partition: {args.partition}")

    logger.info("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path)
    generation_config = {
        "max_new_tokens": args.max_new_tokens,
        "num_beams": args.num_beams,
        "num_beam_groups": args.num_beam_groups,
        "diversity_penalty": args.diversity_penalty,
        "early_stopping": args.early_stopping,
        "num_return_sequences": args.num_return_sequences,
    }

    constrained_function = None
    if args.use_constrained_decoding:
        logger.info("Setting up constrained decoding with trie...")
        trie = Trie.load(args.trie_path)
        constrained_function = constrained_decoding(
            tokenizer, trie, args.start_token, args.end_token
        )

    logger.info(f"Loading data from {args.partition_file_path}")
    data = DataUtils.load_data(args.partition_file_path)[args.partition]
    logger.info(f"Processing {len(data)} samples in batches of {args.batch_size}")

    # Process data in batches
    for i in tqdm(range(0, len(data), args.batch_size)):
        batch = data[i : i + args.batch_size]
        logger.debug(f"Processing batch {i//args.batch_size + 1}")
        prompts = [
            GEN_PSEUDO_GRAPH_PROMPT.replace("{{claim}}", sample["claim"])
            for sample in batch
        ]

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

    save_folder = os.path.join(args.output_folder_path, args.version)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        logger.info(f"Created output directory: {save_folder}")
    
    save_results = {"settings": vars(args), "data": data}
    file_name = f"factkg_test_with_intermediate_graph_{args.partition}.json"
    output_path = os.path.join(save_folder, file_name)
    logger.info(f"Saving results to {output_path}")
    DataUtils.save_json(save_results, output_path)
    logger.info("Processing completed successfully!")
