from src.openai_utils import BatchUtils
from src.utils import DataUtils

temp = {
    "claim": "He had a successor named John E. Beck as well.",
    "id": "a12c2a5317064c9fa74f16dbffcc01f1",
    "Label": [True],
    "Entity_set": ["John_E._Beck"],
    "Evidence": {"John_E._Beck": [["successor"]]},
    "types": ["coll:model", "existence"],
}

if __name__ == "__main__":
    data = DataUtils.load_data("./data/processed_factkg/factkg_train.json")
    print(data[100])
