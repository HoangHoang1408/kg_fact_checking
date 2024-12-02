import os

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential


@retry(
    wait=wait_random_exponential(
        min=float(os.environ["OPENAI_REQUEST_RETRY_TIME_MIN"]),
        max=float(os.environ["OPENAI_REQUEST_RETRY_TIME_MAX"]),
    ),
    stop=stop_after_attempt(int(os.environ["OPENAI_REQUEST_RETRY_MAX_TIMES"])),
)
def get_completion(
    prompt,
    system_prompt=None,
    model=os.environ["OPENAI_CHAT_MODEL"],
    temperature=float(os.environ["OPENAI_GENERATION_TEMPERATURE"]),
    top_p=float(os.environ["OPENAI_GENERATION_TOP_P"]),
    max_tokens=int(os.environ["OPENAI_GENERATION_MAX_TOKENS"]),
):
    client = OpenAI()
    messages = []
    if system_prompt is not None:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content
