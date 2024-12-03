from src.utils import DataUtils
from src.openai_utils import BatchUtils
import argparse
import os

PROMPT = """
Specify if the following entities are mentioned in the claim or not.
Respond correctly in the following JSON format and do not output anything else:
{
    "in_entities": [list of entities that are in the claim],
    "out_entities": [list of entities that are not in the claim]
}
Do not change the entity names from the list of provided entities.
### Claim: {{claim}}
### Entities: {{entities}}
""".strip()


if __name__ == "__main__":
    data = DataUtils.load_data("./data/processed_factkg/factkg_train.json")
    messages = []
    for sample in data:
        messages.append(
            {
                "id": sample["id"],
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
    output_folder = "./openai_batch/batches"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    BatchUtils.prepare_jsonl_for_batch_completions(
        messages,
        file_name="batch_annotate_in_out_entities.jsonl",
        output_folder=output_folder,
        model="gpt-4o-mini",
        temperature=0,
        max_tokens=512,
    )
    print("Done process batch annotate in out entities")
