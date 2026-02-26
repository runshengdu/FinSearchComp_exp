import copy
import random
import time
from typing import Any, Dict, List, Optional

from openai import OpenAI


def _add_prompt_caching(messages: List[Dict[str, Any]], model_name: str) -> List[Dict[str, Any]]:
    if not ("minimax" in model_name.lower() or "claude" in model_name.lower()):
        return messages

    cached_messages = copy.deepcopy(messages)

    for n in range(len(cached_messages)):
        if n < len(cached_messages) - 4:
            continue
        msg = cached_messages[n]
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")

        if isinstance(content, str):
            msg["content"] = [
                {
                    "type": "text",
                    "text": content,
                    "cache_control": {"type": "ephemeral"},
                }
            ]
            continue

        if isinstance(content, list):
            for content_item in content:
                if isinstance(content_item, dict) and "type" in content_item:
                    content_item["cache_control"] = {"type": "ephemeral"}

    return cached_messages


def _call_llm_with_retries(
    client: OpenAI,
    model: str,
    messages: List[Dict[str, Any]],
    tools: List[Dict[str, Any]],
    temperature: Optional[float],
    max_tokens: Optional[int],
    extra_body: Optional[Dict[str, Any]],
    tool_choice: Optional[Any],
) -> Dict[str, Any]:
    last_err: Optional[BaseException] = None
    for attempt in range(1, 4):
        try:
            sanitized_messages: List[Dict[str, Any]] = []
            for m in messages:
                if not isinstance(m, dict) or "usage" not in m:
                    sanitized_messages.append(m)
                    continue
                nm = m.copy()
                nm.pop("usage", None)
                sanitized_messages.append(nm)

            cached_messages = _add_prompt_caching(sanitized_messages, model)
            resp = client.chat.completions.create(
                model=model,
                messages=cached_messages,
                tools=tools,
                tool_choice=tool_choice,
                temperature=temperature,
                max_tokens=max_tokens,
                extra_body=extra_body,
            )
            return resp.model_dump()
        except BaseException as e:
            last_err = e
            if attempt >= 3:
                raise
            time.sleep(0.8 * (2 ** (attempt - 1)) + random.random() * 0.2)
    raise last_err or RuntimeError("LLM 调用失败")
