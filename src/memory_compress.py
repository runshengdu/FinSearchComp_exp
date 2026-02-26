import json
from typing import Any, Dict, List, Tuple

import tiktoken

from utils import json_dumps


_PLACEHOLDER = "tool call success, but tool call results were removed from message history to reduce token usage."

_ENC: Any = None


def _get_encoding() -> Any:
    global _ENC
    if _ENC is None:
        _ENC = tiktoken.get_encoding("cl100k_base")
    return _ENC


def _count_tokens(text: str) -> int:
    enc = _get_encoding()
    return len(enc.encode(text))


def _truncate_text(text: str, max_tokens: int) -> str:
    if max_tokens <= 0:
        return ""
    enc = _get_encoding()
    return enc.decode(enc.encode(text)[: int(max_tokens)])


def _last_assistant_total_tokens(messages: List[Dict[str, Any]]) -> int:
    for msg in reversed(messages):
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        usage = msg.get("usage")
        if not isinstance(usage, dict):
            continue
        total = usage.get("total_tokens")
        if isinstance(total, int) and total >= 0:
            return total
    return 0


def _recent_tool_call_ids_from_assistants(
    messages: List[Dict[str, Any]],
    *,
    upto: int,
    k: int,
) -> set:
    if k <= 0:
        return set()
    upto = min(max(0, int(upto)), len(messages))

    ids: List[str] = []
    selected = 0
    for msg in reversed(messages[:upto]):
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        tool_calls = msg.get("tool_calls")
        if not (isinstance(tool_calls, list) and tool_calls):
            continue
        selected += 1
        for tc in tool_calls:
            if not isinstance(tc, dict):
                continue
            tc_id = tc.get("id")
            if isinstance(tc_id, str) and tc_id:
                ids.append(tc_id)
        if selected >= k:
            break
    return set(ids)

"""first layer of memory compression: remove tool call results from history messages"""
def remove_tool_call_results_from_messages(
    messages: List[Dict[str, Any]],
    *,
    seen_upto: int,
    keep_last_k: int = 5,
    context_window: int,
    threshold_ratio: float = 0.85,
    enabled: bool = False,
) -> Tuple[List[Dict[str, Any]], bool]:
    if not isinstance(context_window, int) or context_window <= 0:
        raise RuntimeError(f"context_window 非法: {context_window!r}")
    should_reduce = enabled
    if not should_reduce:
        total_tokens = _last_assistant_total_tokens(messages)
        should_reduce = total_tokens >= int(float(threshold_ratio) * int(context_window))

    if not should_reduce:
        return messages, False

    if seen_upto <= 0:
        return messages, True
    if seen_upto > len(messages):
        seen_upto = len(messages)

    k = int(keep_last_k) if keep_last_k is not None else 0
    if k < 0:
        k = 0
    keep_tool_call_ids = _recent_tool_call_ids_from_assistants(messages, upto=seen_upto, k=k)

    reduced: List[Dict[str, Any]] = []
    for i, msg in enumerate(messages):
        if i >= seen_upto or not isinstance(msg, dict) or msg.get("role") != "tool":
            reduced.append(msg)
            continue

        tool_call_id = msg.get("tool_call_id")
        if isinstance(tool_call_id, str) and tool_call_id in keep_tool_call_ids:
            reduced.append(msg)
            continue

        new_msg = msg.copy()
        new_msg["content"] = _PLACEHOLDER
        reduced.append(new_msg)

    return reduced, True

"""second layer of memory compression: compress web content tool call results from history messages"""
def compress_web_content(
    messages: List[Dict[str, Any]],
    *,
    max_text_tokens: int = 3000,
) -> Tuple[List[Dict[str, Any]], bool]:
    if not isinstance(max_text_tokens, int) or max_text_tokens <= 0:
        raise RuntimeError(f"max_text_tokens 非法: {max_text_tokens!r}")

    try:
        import web_summary
    except Exception:
        return messages, False

    web_content_tool_call_ids: set = set()
    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        tool_calls = msg.get("tool_calls")
        if not isinstance(tool_calls, list):
            continue
        for tc in tool_calls:
            if not isinstance(tc, dict):
                continue
            tc_id = tc.get("id")
            fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
            fn_name = fn.get("name")
            if fn_name != "web_content":
                continue
            if isinstance(tc_id, str) and tc_id:
                web_content_tool_call_ids.add(tc_id)

    if not web_content_tool_call_ids:
        return messages, False

    changed = False
    out: List[Dict[str, Any]] = []
    for msg in messages:
        if (
            not isinstance(msg, dict)
            or msg.get("role") != "tool"
            or msg.get("tool_call_id") not in web_content_tool_call_ids
        ):
            out.append(msg)
            continue

        content = msg.get("content")
        if not isinstance(content, str) or not content:
            out.append(msg)
            continue

        if content == _PLACEHOLDER:
            out.append(msg)
            continue

        stripped = content.lstrip()
        if not stripped or stripped[0] not in "{[":
            out.append(msg)
            continue

        try:
            tool_result = json.loads(content)
        except Exception:
            out.append(msg)
            continue

        if not isinstance(tool_result, dict) or tool_result.get("error") is not None:
            out.append(msg)
            continue

        results = tool_result.get("results")
        if not isinstance(results, list) or not results:
            out.append(msg)
            continue

        local_changed = False
        for item in results:
            if not isinstance(item, dict):
                continue
            text = item.get("text")
            if not isinstance(text, str) or not text:
                continue
            if _count_tokens(text) <= int(max_text_tokens):
                continue
            try:
                summarized = web_summary.summarize_text(text)
                if isinstance(summarized, str) and summarized.strip():
                    item["text"] = summarized
                    local_changed = True
            except Exception:
                continue

        if not local_changed:
            out.append(msg)
            continue

        new_msg = msg.copy()
        new_msg["content"] = json_dumps(tool_result)
        out.append(new_msg)
        changed = True

    return out, changed

"""third layer of memory compression: apply web summary to web_content tool call results"""
def apply_web_summary(
    tool_result: Any,
    *,
    context_window: int,
    last_assistant_total_tokens: int,
) -> Any:
    if not isinstance(context_window, int) or context_window <= 0:
        raise RuntimeError(f"context_window 未设置或非法: {context_window!r}")
    if not isinstance(tool_result, dict):
        return tool_result
    if tool_result.get("error") is not None:
        return tool_result

    results = tool_result.get("results")
    if not isinstance(results, list) or not results:
        return tool_result

    threshold = int(0.5 * context_window)
    for item in results:
        if not isinstance(item, dict):
            continue
        text = item.get("text")
        if not isinstance(text, str) or not text:
            continue
        text_tokens = _count_tokens(text)
        if text_tokens + int(last_assistant_total_tokens) <= threshold:
            continue
        try:
            import web_summary

            summarized = web_summary.summarize_text(text)
            if isinstance(summarized, str) and summarized.strip():
                item["text"] = summarized
        except Exception:
            budget = int(threshold) - int(last_assistant_total_tokens)
            if budget <= 0:
                budget = max(1, min(256, threshold // 10))
            item["text"] = _truncate_text(text, budget)

    return tool_result
