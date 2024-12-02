def extract_openai_result(openai_result):
    return openai_result["response"]["body"]["choices"][0]["message"]["content"]
