import sys
from openai import OpenAI
import os
from llm import _call_llm_with_retries


def summarize_text(text: str, model_config_name: str = "qwen3.5-plus") -> str:
    messages = [
        {
            "role": "system",
            "content": "You summarize long web page text for downstream QA. Preserve key facts, dates, numbers, names, and definitions. Write in the same language as the input. your summary should be as detailed as possible.",
        },
        {"role": "user", "content": text},
    ]

    api_key = os.getenv("DASHSCOPE_API_KEY")
    client = OpenAI(base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", api_key=api_key)
    resp = _call_llm_with_retries(
        client=client,
        model=model_config_name,
        messages=messages,
        tools=[],
        temperature=None,
        max_tokens=None,
        extra_body=None,
        tool_choice="none",
    )
    choice = (resp.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    content = message.get("content") or ""
    return content if isinstance(content, str) else ""


if __name__ == "__main__":
    raw = sys.stdin.read()
    sys.stdout.write(summarize_text(raw))
