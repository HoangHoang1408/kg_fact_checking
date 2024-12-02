from utils import LoadData
from openai_utils import BatchUtils
import argparse
import os


PROMPT = """
Specify if the following entities are mentioned in the claim or not.
Respond correctly in the following JSON format and do not output anything else:
{
    "in_entities": [entity1, entity2],
    "out_entities": [entity1, entity2]
}

### Claim: {{claim}}
### Entities: {{entities}}
""".strip()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file-path", type=str, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    data = LoadData.load_data(args.input_file_path)
    data = [{"claim": k, **v} for k, v in data.items()]
    messages = []
    for i, sample in enumerate(data):
        messages.append(
            {
                "id": i,
                "messages": [
                    {
                        "role": "user",
                        "content": PROMPT.replace("{{claim}}", sample["claim"]).replace(
                            "{{entities}}", str(sample["Entity_set"])
                        ),
                    }
                ],
            }
        )
    BatchUtils.prepare_jsonl_for_batch_completions(
        messages,
        file_name="batch_annotate_in_out_entities.jsonl",
        output_folder="./openai_batch_jobs",
        model="gpt-4o-mini",
        temperature=0,
        max_tokens=512,
    )
    print("Done")
