import json
from typing import Any, Dict, List


def _reduce_tool_message_content_drop_text(content: str) -> str:
    try:
        data = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return content

    if not isinstance(data, dict):
        return content

    results = data.get("results")
    if not isinstance(results, list):
        return content

    modified = False
    for item in results:
        if isinstance(item, dict) and "text" in item:
            del item["text"]
            modified = True

    if not modified:
        return content
    return json.dumps(data, ensure_ascii=False, indent=2)


def _tool_call_id_to_name_map(messages: List[Dict[str, Any]]) -> Dict[str, str]:
    tool_call_id_to_name: Dict[str, str] = {}
    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        tool_calls = msg.get("tool_calls")
        if not isinstance(tool_calls, list):
            continue
        for tc in tool_calls:
            if not isinstance(tc, dict):
                continue
            tool_call_id = tc.get("id")
            fn = tc.get("function")
            if not isinstance(tool_call_id, str) or not isinstance(fn, dict):
                continue
            tool_name = fn.get("name")
            if isinstance(tool_name, str):
                tool_call_id_to_name[tool_call_id] = tool_name
    return tool_call_id_to_name


def _apply_seen_tool_text_reduction(
    messages: List[Dict[str, Any]],
    *,
    seen_upto: int,
    tool_name: str,
) -> List[Dict[str, Any]]:
    if seen_upto <= 0:
        return messages
    if seen_upto > len(messages):
        seen_upto = len(messages)

    tool_call_id_to_name = _tool_call_id_to_name_map(messages)
    reduced: List[Dict[str, Any]] = []
    for i, msg in enumerate(messages):
        if i >= seen_upto or not isinstance(msg, dict) or msg.get("role") != "tool":
            reduced.append(msg)
            continue
        tool_call_id = msg.get("tool_call_id")
        if not isinstance(tool_call_id, str) or tool_call_id_to_name.get(tool_call_id) != tool_name:
            reduced.append(msg)
            continue
        content = msg.get("content")
        if not isinstance(content, str):
            reduced.append(msg)
            continue
        new_msg = msg.copy()
        new_msg["content"] = _reduce_tool_message_content_drop_text(content)
        reduced.append(new_msg)
    return reduced

