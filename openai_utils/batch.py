import os

from datasets import Dataset
from openai import OpenAI


class BatchUtils:
    @staticmethod
    def upload_file(jsonl_file_path):
        client = OpenAI()
        return client.files.create(file=open(jsonl_file_path, "rb"), purpose="batch")

    @staticmethod
    def submit_batch(file_id, metadata={"description": "a batch job"}):
        client = OpenAI()
        return client.batches.create(
            input_file_id=file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata=metadata,
        )

    @staticmethod
    def check_status(batch_id):
        client = OpenAI()
        return client.batches.retrieve(batch_id)

    @staticmethod
    def list_batches():
        client = OpenAI()
        return client.batches.list()

    @staticmethod
    def cancel_batch(batch_id):
        client = OpenAI()
        return client.batches.cancel(batch_id)

    @staticmethod
    def retrieve_batch(file_id):
        client = OpenAI()
        return client.files.content(file_id)

    @staticmethod
    def prepare_jsonl_for_batch_completions(
        messages_with_ids,
        file_name="batch_completions.jsonl",
        output_folder=".",
        model=os.environ["OPENAI_CHAT_MODEL"],
        temperature=float(os.environ["OPENAI_GENERATION_TEMPERATURE"]),
        top_p=float(os.environ["OPENAI_GENERATION_TOP_P"]),
        max_tokens=int(os.environ["OPENAI_GENERATION_MAX_TOKENS"]),
    ):
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        output_path = os.path.join(output_folder, file_name)
        data = []
        for message in messages_with_ids:
            data.append(
                {
                    "custom_id": message["id"],
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": model,
                        "messages": message["messages"],
                        "temperature": temperature,
                        "top_p": top_p,
                        "max_tokens": max_tokens,
                    },
                }
            )
        Dataset.from_list(data).to_json(output_path, force_ascii=False)
