import os
import asyncio
import logging
import threading
import time
from typing import Any, Dict, List
import requests
import tiktoken
from src import content_summary
from utils import json_dumps

logger = logging.getLogger(__name__)

_TOOL_MIN_INTERVAL_SECONDS: Dict[str, float] = {
    "web_search_chinese": 0.2,
    "web_search_global": 0.2,
    "web_content": 0.1,
}
_TOOL_LAST_CALL_TS: Dict[str, float] = {k: 0.0 for k in _TOOL_MIN_INTERVAL_SECONDS}
_TOOL_LOCKS: Dict[str, threading.Lock] = {k: threading.Lock() for k in _TOOL_MIN_INTERVAL_SECONDS}

_TOOL_MAX_CONCURRENCY: Dict[str, int] = {
    "web_search_chinese": 40,
}
_TOOL_SEMAPHORES: Dict[str, threading.BoundedSemaphore] = {
    k: threading.BoundedSemaphore(v) for k, v in _TOOL_MAX_CONCURRENCY.items() if v and v > 0
}

def _count_tokens(text: str) -> int:
    encoding = tiktoken.encoding_for_model("cl100k_base")
    return len(encoding.encode(text))

def _truncate_text(text:str, max_tokens:int) -> str:
    enc=tiktoken.get_encoding("cl100k_base")
    return enc.decode(enc.encode(text)[:max_tokens])

def _tool_rate_limit(tool_name: str) -> None:
    min_interval = float(_TOOL_MIN_INTERVAL_SECONDS.get(tool_name, 0.0) or 0.0)
    if min_interval <= 0:
        return
    lock = _TOOL_LOCKS.get(tool_name)
    if lock is None:
        return
    with lock:
        now = time.monotonic()
        last = float(_TOOL_LAST_CALL_TS.get(tool_name, 0.0) or 0.0)
        wait_s = (last + min_interval) - now
        if wait_s > 0:
            time.sleep(wait_s)
        _TOOL_LAST_CALL_TS[tool_name] = time.monotonic()

def _make_api_request(
    url: str,
    headers: Dict[str, str],
    payload: Dict[str, Any],
    timeout: int = 60
) -> Dict[str, Any]:
    resp = requests.post(
        url,
        headers=headers,
        data=json_dumps(payload).encode("utf-8"),
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()


def web_search_chinese(query: str, count: int = 5) -> Dict[str, Any]:
    _tool_rate_limit("web_search_chinese")
    api_key = os.environ.get("GLM_API_KEY")
    sem = _TOOL_SEMAPHORES.get("web_search_chinese")
    if sem is None:
        data = _make_api_request(
            "https://open.bigmodel.cn/api/paas/v4/web_search",
            {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            {
                "search_query": query,
                "search_engine": "search_std",
                "count": int(count),
                "content_size": "medium",
                "search_intent": False,
            },
            timeout=60
        )
    else:
        sem.acquire()
        try:
            data = _make_api_request(
                "https://open.bigmodel.cn/api/paas/v4/web_search",
                {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                {
                    "search_query": query,
                    "search_engine": "search_std",
                    "count": int(count),
                    "content_size": "medium",
                    "search_intent": False,
                },
                timeout=60
            )
        finally:
            sem.release()

    results = data.get("search_result") or data.get("search_results") or []
    compact = []
    if isinstance(results, list):
        for r in results[: int(count)]:
            if not isinstance(r, dict):
                continue
            compact.append(
                {
                    "title": r.get("title"),
                    "link": r.get("link"),
                    "content": r.get("content"),
                    "media": r.get("media"),
                    "publish_date": r.get("publish_date"),
                }
            )
    return {"query": query, "results": compact}


def web_search_global(query: str, max_results: int = 5) -> Dict[str, Any]:
    _tool_rate_limit("web_search_global")
    api_key = os.environ.get("PARALLEL_API_KEY")
    data = _make_api_request(
        "https://api.parallel.ai/v1beta/search",
        {
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "parallel-beta": "search-extract-2025-10-10",
        },
        {
            "mode": "one-shot",
            "objective": query,
            "search_queries": [query],
            "max_results": int(max_results),
            "excerpts": {"max_chars_per_result": 1000},
        },
        timeout=90
    )

    results = data.get("results") or []
    compact = []
    if isinstance(results, list):
        for r in results[: int(max_results)]:
            if not isinstance(r, dict):
                continue
            excerpts = r.get("excerpts")
            compact.append(
                {
                    "title": r.get("title"),
                    "url": r.get("url"),
                    "publish_date": r.get("publish_date"),
                    "excerpts": excerpts if isinstance(excerpts, list) else [],
                }
            )
    return {"query": query, "results": compact}


def web_content(url: str, text: bool = True, summary: bool = False) -> Dict[str, Any]:
    _tool_rate_limit("web_content")
    api_key = os.environ.get("EXA_API_KEY")
    payload: Dict[str, Any] = {"urls": [url]}
    payload["text"] = True
    payload["summary"] = False

    data = _make_api_request(
        "https://api.exa.ai/contents",
        {"Content-Type": "application/json", "x-api-key": api_key},
        payload,
        timeout=120
    )

    results = data.get("results") or []
    compact = []
    max_text_tokens=5000
    if isinstance(results, list):
        for r in results[:1]:
            if not isinstance(r, dict):
                continue
            text_content = r.get("text")
            processed_text = text_content
            if _count_tokens(text_content) > max_text_tokens:
                try:
                    processed_text = content_summary.summarize_text(text_content)
                except Exception as e:
                    processed_text = _truncate_text(text_content, max_text_tokens)
                    logger.error(f"Failed to summarize text for {url}: {e}")
            compact.append(
                {
                    "title": r.get("title"),
                    "url": r.get("url"),
                    "publishedDate": r.get("publishedDate"),
                    "text": processed_text,
                    "summary": r.get("summary"),
                }
            )
    return {"results": compact, "statuses": data.get("statuses")}


def tool_specs() -> List[Dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "web_search_chinese",
                "description": "当搜索关键词为中文时，用此工具进行联网搜索，返回标题/链接/摘要等信息。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "count": {"type": "integer", "default": 10, "minimum": 1, "maximum": 50},
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "web_search_global",
                "description": "当搜索关键词不是中文时，用此工具进行联网搜索，返回 URL/摘录等信息。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "max_results": {"type": "integer", "default": 10, "minimum": 1, "maximum": 20},
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "web_content",
                "description": "读取指定网页的正文内容（可选摘要），用于进一步核对与引用。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string"},
                        "text": {"type": "boolean", "default": True},
                        "summary": {"type": "boolean", "default": False},
                    },
                    "required": ["url"],
                },
            },
        },
    ]


TOOL_FN_MAP = {
    "web_search_chinese": web_search_chinese,
    "web_search_global": web_search_global,
    "web_content": web_content,
}
