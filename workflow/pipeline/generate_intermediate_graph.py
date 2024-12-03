from src.hf_utils import generate
from src.utils import DataUtils
from argparse import ArgumentParser
from transformers import AutoTokenizer, AutoModelForCausalLM


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--data-folder-path", type=str, required=True)
    parser.add_argument("--output-folder-path", type=str, required=True)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--num-beams", type=int, default=5)
    parser.add_argument("--num-beam-groups", type=int, default=1)
    parser.add_argument("--diversity-penalty", type=float, default=0.0)
    parser.add_argument("--early-stopping", type=bool, default=True)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path)
    
    dev_data = DataUtils.load_data(
        os.path.join(args.data_folder_path, "processed_factkg", "factkg_dev.json")
    )
    
    generate(dev_data, args.output_folder_path, tokenizer, model, **vars(args))
    generate(test_data, args.output_folder_path, tokenizer, model, **vars(args))
